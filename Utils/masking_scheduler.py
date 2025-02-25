import math
import torch


def get_mask_code(code, r=None, mode="arccos", value=None, **kargs):
    """ Replace the code token by *value* according the the *mode* scheduler
       :param
        code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
        mode  -> str:                the rate of value to mask
        value -> int:                mask the code by the value
       :return
        masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
        mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
    """
    b, h, w = code.size()
    device = code.device
    if r is None:
        r = torch.rand(b)
    if mode == "root":      # root scheduler
        val_to_mask = 1 - (r ** .5)
    elif mode == "linear":  # linear scheduler
        val_to_mask = 1 - r
    elif mode == "square":  # square scheduler
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":  # cosine scheduler
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":  # arc cosine scheduler
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    else:
        return

    mask_code = code.detach().clone()
    # Sample the number of tokens + localization to mask
    mask = torch.rand(size=code.size()).to(device) < val_to_mask.view(b, 1, 1).to(device)

    if value > 0:  # Mask the selected token by the value
        mask_code[mask] = torch.full_like(mask_code[mask], value)

    # reconstruct every token, nothing is masked
    loss_mask = torch.ones_like(code).bool()

    return mask_code, mask, loss_mask

