import torch
import gradio as gr
import numpy as np
from PIL import Image
from Utils.utils import load_args_from_file
from huggingface_hub import hf_hub_download
import torchvision.utils as vutils
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler


# Load config and set device
config_path = "Config/base_cls2img.yaml"
args = load_args_from_file(config_path)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download and load pre-trained models
hf_hub_download(repo_id="FoundationVision/LlamaGen", filename="vq_ds16_c2i.pt", local_dir="./saved_networks/")
hf_hub_download(repo_id="llvictorll/Halton-Maskgit", filename="ImageNet_384_large.pth", local_dir="./saved_networks/")

# Initialize the model
model = MaskGIT(args)


# Function for Gradio
def generate_images(label_input, sm_temp_min, sm_temp_max, temp_pow, temp_warmup, w, sched_pow, step, randomize, top_k):

    labels = list(map(int, label_input.split(",")))  # Convert input to list of integers
    labels = torch.LongTensor(labels).to(args.device)

    # Define sampler with user-provided values
    sampler = HaltonSampler(
        sm_temp_min=sm_temp_min,
        sm_temp_max=sm_temp_max,
        temp_pow=temp_pow,
        temp_warmup=temp_warmup,
        w=w,
        sched_pow=sched_pow,
        step=step,
        randomize=randomize,
        top_k=top_k
    )

    gen_images = sampler(trainer=model, nb_sample=len(labels), labels=labels, verbose=True)[0]
    min_norm = gen_images.min(-1)[0].min(-1)[0].min(-1)[0].view(-1, 1, 1, 1)
    max_norm = gen_images.max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1)
    gen_images = (gen_images - min_norm) / (max_norm - min_norm)
    gen_images = vutils.make_grid(gen_images.cpu(), normalize=True).permute(1, 2, 0).numpy() # Convert to PIL for Gradio
    img = Image.fromarray((gen_images * 255).astype(np.uint8))
    return img


# Gradio Interface
interface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox("1,7,282", label="Enter class labels (comma-separated, e.g., 1,7,282)"),
        gr.Slider(0.1, 2.0, 1.0, step=0.1, label="sm_temp_min"),
        gr.Slider(0.1, 2.0, 1.2, step=0.1, label="sm_temp_max"),
        gr.Slider(0.1, 5.0, 1.0, step=0.1, label="temp_pow"),
        gr.Slider(0, 100, 0, step=1, label="temp_warmup"),
        gr.Slider(1, 10, 2, step=1, label="w"),
        gr.Slider(1, 5, 2, step=0.1, label="sched_pow"),
        gr.Slider(1, 100, 32, step=1, label="step"),
        gr.Checkbox(True, label="randomize"),
        gr.Slider(-1, 16384, -1, step=1, label="top_k")
    ],
    outputs=gr.Image(type="pil"),
    title="Halton MaskGIT Image Generator",
    description="Generate images using the Halton MaskGIT model. Adjust sampling parameters for more control over the output.",
)

# Launch Gradio app
if __name__ == "__main__":
    interface.launch(server_port=6006)
