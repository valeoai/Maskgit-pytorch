import torch
from Network.Taming.modules.vqvae.latent_quantize import LatentQuantize

# Test case 1: Test forward pass
n_e = 3
edim = 8
commitment_loss_weight = 0.1
quantization_loss_weight = 0.1
optimize_values = True

model = LatentQuantize(n_e, edim, commitment_loss_weight, quantization_loss_weight, optimize_values)
input_tensor = torch.randn(4, edim, 16, 16)  # Batch size of 4, edim of 32, spatial dimensions of 16x16
output_codes, loss, _ = model(input_tensor)
assert output_codes.shape == input_tensor.shape  # Output codes should have the same shape as the input

# Test case 2: Test quantization loss
z = torch.randn(4, edim, 16, 16)
zhat = torch.randn(4, edim, 16, 16)
reduce = "mean"

quantization_loss = model.quantization_loss(z, zhat, reduce)
assert quantization_loss.item() >= 0  # Quantization loss should be non-negative

# Test case 3: Test commitment loss
commitment_loss = model.commitment_loss(z, zhat, reduce)
assert commitment_loss.item() >= 0  # Commitment loss should be non-negative

# Test case 4: Test quantize function
z = torch.randn(4, 16, 16, edim)
quantized_z = model.quantize(z)
assert quantized_z.shape == z.shape  # Quantized z should have the same shape as z

print("All tests passed!")