import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange




from typing import List, Optional, Union, Callable
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor, int32
from torch.optim import Optimizer

from einops import rearrange, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# main class

class LatentQuantize(Module):
    def __init__(
            self,
            n_e: int,
            edim: int,
            commitment_loss_weight: Optional[float] = 0.1,
            quantization_loss_weight: Optional[float] = 0.1,
            optimize_values: Optional[bool] = True,
        ):
            '''
            Initializes the LatentQuantization module.

            Args:
                n_e (int): number of embeddings per dim
                edim (int): dimension of embedding
                beta (float): commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
                optimize_values (Optional[bool]): Whether to optimize the values of the codebook. If not provided, it is set to True.
            '''
            super().__init__()

            self.register_buffer("commitment_loss_weight", torch.tensor(commitment_loss_weight, dtype=torch.float32), persistent = False)
            self.register_buffer("quantization_loss_weight", torch.tensor(quantization_loss_weight, dtype=torch.float32), persistent = False)
            self.register_buffer("edim",  torch.tensor(edim), persistent = False) 
            self.register_buffer("n_e", torch.tensor(n_e), persistent = False)
            self.register_buffer("_levels", torch.tensor([n_e,]*edim), persistent = False)
            
            _basis =   torch.cumprod(torch.concat([torch.tensor([1], dtype=int32), self._levels[:-1]], dim=0), dim=0)
            self.register_buffer("_basis", _basis, persistent = False)

            self.codebook_size = n_e**edim

            if self.codebook_size > 2**16 or self.codebook_size < 0:
                print("Warning: codebook size is larger than 2**16, which is not supported by the current implementation. Using lookup-free quantized latents instead.")
                self._codebook = None
            else:
                self._codebook = self.indices_to_codes(torch.arange(self.codebook_size))

            values_per_latent = [torch.linspace(-0.5, 0.5, n_e) if n_e%2==1 else torch.arange(n_e)/n_e - 0.5 for _ in range(edim)] #ensure zero is in the middle and start is always -0.5
            values_per_latent = torch.stack(values_per_latent, dim=0)
            
            #test, and check whether it would be in the parameters of the model or not
            if optimize_values:
                self.values_per_latent = nn.Parameter(values_per_latent) 
            else:
                self.register_buffer('values_per_latent', values_per_latent)

    def quantization_loss(self, z: Tensor, zhat: Tensor, reduce="mean") -> Tensor:
        """Computes the quantization loss."""
        return F.mse_loss(zhat.detach(), z, reduction=reduce)

    def commitment_loss(self, z: Tensor, zhat: Tensor, reduce="mean") -> Tensor:
        """Computes the commitment loss."""
        return F.mse_loss(z.detach(), zhat, reduction=reduce)    

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z.
        The quantization is done by measuring the distance between the input and the codebook values per latent dimension
        and returning the index of the closest codebook value.
        """
        def distance(x, y):
            return torch.abs(x - y)
        

        index = torch.argmin(distance(z[..., None], self.values_per_latent), dim=-1)
        quantize = self.values_per_latent[torch.arange(self.edim), index]
        quantize = z + (quantize - z).detach()
        return quantize 
    
    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        """ scale and shift zhat from [-0.5, 0.5] to [0, level_per_dim]"""
        half_width = self._levels // 2
        return (zhat_normalized * 2 * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        """normalize zhat to [-0.5, 0.5]"""
        half_width = self._levels // 2
        return (zhat - half_width) / half_width / 2
    
    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` which contains the number per latent to an index in the codebook."""
        assert zhat.shape[-1] == self.edim, f'expected dimension of {self.edim} but found dimension of {zhat.shape[-1]}'
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)
    
    def codes_to_indices_per_latent(self, zhat: Tensor) -> Tensor:
        """Converts a `code` which contains the number per latent to an index in the codebook."""
        assert zhat.shape[-1] == self.edim, f'expected dimension of {self.edim} but found dimension of {zhat.shape[-1]}'
        zhat = self._scale_and_shift(zhat)
        return zhat.to(int32)
    

    def indices_to_codes(
        self,
        indices: Tensor
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= 3

        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)


        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self,
                 z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension 
        c - number of codebook dim
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (b, d, h, w)
        quantization pipeline:
            1. get encoder input (b,d,h,w)
            2. flatten input to (B*H*W,D)
        """

        is_img_or_video = z.ndim >= 4
        original_input_flatten = z.view(-1, self.edim)

        # standardize image or video into (batch, seq, dimension)
        if is_img_or_video:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.edim, f'expected dimension of {self.edim} but found dimension of {z.shape[-1]}'

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        #min_encodings = torch.zeros(*indices.shape, self.codebook_size).to(indices)
        #min_encodings.scatter_(-1, indices.unsqueeze(-1).to(torch.int64), 1)
        min_encodings = None # waste of memory, we don't need it

        # reconstitute image or video dimensions
        if is_img_or_video:
            out = unpack_one(codes, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')
            indices = unpack_one(indices, ps, 'b *')
            #min_encodings = unpack_one(min_encodings, ps, 'b * ...')

        #get one hot encoding from indices


        #calculate losses
        commitment_loss = self.commitment_loss(original_input_flatten, codes.view(-1, self.edim)) if self.training and self.commitment_loss_weight!=0  else torch.tensor(0.)
        quantization_loss = self.quantization_loss(original_input_flatten, codes.view(-1, self.edim)) if self.training and self.quantization_loss_weight!=0 else torch.tensor(0.)


        loss = self.commitment_loss_weight * commitment_loss + self.quantization_loss_weight * quantization_loss 
        #z_q, loss, (perplexity, min_encodings, min_encoding_indices)
        #return out, indices, loss
        return out, loss, (None, min_encodings, indices)
