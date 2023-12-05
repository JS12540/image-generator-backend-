from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import mediapy as media
from torch import autocast
from src.model import process_images


app = FastAPI()

# CORS (Cross-Origin Resource Sharing) settings to allow frontend access
origins = [
    "http://localhost",
    "http://localhost:8000",  # Add your other allowed origins here,
    "https://image-generation-frontend-dp4i9cdmx-jay-shahs-projects.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class InputData(BaseModel):
    prompt: str
    num_images: int

class OutputData(BaseModel):
    image_data: str

@app.post("/generate_image", response_model=OutputData)
async def generate_image(data: InputData):
    # Your existing code for image generation
    prompt = data.prompt
    num_images = data.num_images

    # Call the modified process_images function
    image_data = process_images(prompt, num_images)

    return OutputData(image_data=image_data)

