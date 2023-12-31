# -*- coding: utf-8 -*-
"""socialposts_gan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10jiBWktJZgJGtllZFcmpiiWEjP2IxRAy
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install --quiet --upgrade diffusers transformers scipy mediapy accelerate bitsandbytes

from diffusers import PNDMScheduler, DDIMScheduler, LMSDiscreteScheduler

scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True)

import torch
import mediapy as media
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")

prompt = "Generate a stoicism-inspired social media image with a stoic quote. Emphasize simplicity and wisdom. Utilize muted tones and a calm aesthetic to evoke contemplation. Overlay the quote in an elegant font, ensuring readability. Focus on conveying resilience and inner strength visually. Optimize for a square format suitable for various platforms. Prioritize a cohesive design that resonates with stoic principles. Keep the image uncluttered, promoting a sense of tranquility for effective engagement on our stoic-themed social media."
num_images = 3
prompts = [ prompt ] * num_images

with autocast("cuda"):
  images = pipe(prompts, guidance_scale=10, num_inference_steps=150).images

media.show_images(images)
images[0].save("output2.jpg")