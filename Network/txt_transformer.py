# BERT architecture for the Masked Bidirectional Encoder Transformer
import torch
from torch import nn


def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(Attention, self).__init__()
        self.dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True)

    def forward(self, x, cond=None):
        if cond is not None:
            x = self.mha(x, cond, cond, need_weights=False)[0]
        else:
            x = self.mha(x, x, x, need_weights=False)[0]
        return x


class NormLayer(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(x_dim, eps=1e-6)

    def forward(self, x):
        x = self.norm_final(x)
        return x


class Block(nn.Module):
    def __init__(self, hidden_dim, cond_dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden_dim),
                                 nn.LayerNorm(hidden_dim))

        self.ln1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attn = Attention(hidden_dim, heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attn = Attention(hidden_dim, heads, dropout=dropout)

        self.ln3 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ff = FeedForward(hidden_dim, mlp_dim, dropout=dropout)

    def forward(self, x, cond):
        cond = self.mlp(cond)
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), cond)
        x = x + self.ff(self.ln3(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, cond_dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for layer in range(depth):
            self.layers.append(
                Block(hidden_dim, cond_dim, heads, mlp_dim, dropout=dropout)
            )

    def forward(self, x, cond):
        for block in self.layers:
            x = block(x, cond)
        return x


class Transformer(nn.Module):
    def __init__(self, input_size=16, c=768, hidden_dim=768, cond_dim=768, codebook_size=1024,
                 depth=12, heads=16, mlp_dim=3072, dropout=0., proj=1):
        super().__init__()

        self.c = c
        self.proj = proj
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size

        self.tok_emb = nn.Embedding(codebook_size+1, c)
        self.pos_emb = nn.Embedding(self.input_size ** 2, c)

        self.first_norm = nn.LayerNorm(c, eps=1e-6)
        self.in_proj = nn.Conv2d(c, hidden_dim, kernel_size=proj, stride=proj, bias=False)

        self.transformer = TransformerEncoder(hidden_dim=hidden_dim, cond_dim=cond_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        self.out_proj = nn.Sequential(nn.Conv2d(hidden_dim, c*(proj**2), kernel_size=1, stride=1, bias=False),
                                      nn.PixelShuffle(proj)
                                      )

        self.last_norm = nn.LayerNorm(c, eps=1e-6)
        self.bias = nn.Parameter(torch.zeros(1, 1, codebook_size+1))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Init embedding
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        # Init projection in
        nn.init.xavier_uniform_(self.in_proj.weight)

        # Zero-out cross attn projection layers in blocks:
        for block in self.transformer.layers:
            nn.init.constant_(block.mlp[0].weight, 0)
            nn.init.constant_(block.mlp[0].bias, 0)

        # Init projection out
        nn.init.xavier_uniform_(self.out_proj[0].weight)

    def partially_init_from_pretrained(self, ckpt):
        pretrained_model = ckpt['model_state_dict']
        print("Copy only transformer weights from pretrained model")
        for source_parameter, target_parameter in zip(pretrained_model.keys(), self.state_dict().keys()):
            if source_parameter == target_parameter and \
                    self.state_dict()[target_parameter].size() == pretrained_model[source_parameter].size()\
                    and "transformer" in source_parameter:
                print("copying:", source_parameter, self.state_dict()[target_parameter].size())
                self.state_dict()[target_parameter].data.copy_(pretrained_model[source_parameter])

    def forward(self, x, y, drop_label=None):
        b, w, h = x.size()

        # Drop the text-label y --> b, seq_len, cond_dim
        if drop_label is not None:
            y = torch.where(drop_label.view(-1, 1, 1), torch.zeros_like(y), y)

        # position embedding
        pos = torch.arange(0, w*h, dtype=torch.long, device=x.device)  # shape (t)
        pos = self.pos_emb(pos)

        x = x.view(b, -1)
        x = self.tok_emb(x) + pos                                            # b, w*h, c

        x = self.first_norm(x)

        # reshape, proj to smaller space, reshape (patchify!)
        x = x.transpose(1, 2).contiguous().view(b, self.c, w, h)                         # b, c, w, h
        x = self.in_proj(x)                                                              # b, hidden, w // proj, h // proj
        x = x.view(b, self.hidden_dim, -1).transpose(1, 2).contiguous()                  # b, (w // proj * h // proj), hidden

        x = self.transformer(x, y)                                                       # b, (w // proj * h // proj), hidden

        x = x.transpose(1, 2).contiguous().view(b, self.hidden_dim, w//self.proj, h//self.proj)  # b, hidden, w // proj, h // proj
        x = self.out_proj(x)                                                               # b, hidden//proj**2, w, h
        x = x.view(b, self.c, -1).transpose(1, 2).contiguous()                             # b, w * h, hidden//proj**2

        x = self.last_norm(x) # normalize before final prediction

        logit = torch.matmul(x, self.tok_emb.weight.T) + self.bias

        return logit


if __name__ == "__main__":
    model = Transformer(input_size=32, c=1024, hidden_dim=1024, cond_dim=2048,
                        codebook_size=16384, depth=24, heads=16, mlp_dim=1024*4,
                        dropout=0., proj=2)

    code = torch.randint(0, 16384, size=(2, 32, 32))
    txt = torch.randn(2, 120, 2048)
    drop_label = (torch.rand(2) < 0.1)

    c = model(x=code, y=txt, drop_label=drop_label)
    print(c.size())
