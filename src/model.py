from diffusers import PNDMScheduler
import torch
import mediapy as media
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from io import BytesIO
import base64

scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True)
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

def process_images(prompt, num_images):
    prompts = [prompt] * num_images

    with autocast("cuda"):
        images = pipe(prompts, guidance_scale=10, num_inference_steps=150).images

    # Convert the image to base64
    image_buffer = BytesIO()
    images[0].save(image_buffer, format='JPEG')
    image_data = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    return image_data