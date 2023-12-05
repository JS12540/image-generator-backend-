from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import mediapy as media
from torch import autocast
from src.model import process_images
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class InputData(BaseModel):
    prompt: str
    num_images: int

class OutputData(BaseModel):
    image_data: str

@app.options("/generate_image")
async def options_generate_image():
    return {"msg": "Preflight request received"}

@app.post("/generate_image", response_model=OutputData)
async def generate_image(data: InputData):
    try:
        # Your existing code for image generation
        prompt = data.prompt
        num_images = data.num_images

        # Log input data
        logger.info(f"Received request with prompt: {prompt} and num_images: {num_images}")

        # Call the modified process_images function
        image_data = process_images(prompt, num_images)

        # Log success
        logger.info("Image generation successful")

        return OutputData(image_data=image_data)
    except Exception as e:
        # Log the exception
        logger.error(f"Error during image generation: {str(e)}")
        # Raise HTTPException with 500 status code and error detail
        raise HTTPException(status_code=500, detail=str(e))
