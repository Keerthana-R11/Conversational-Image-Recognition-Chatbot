# Conversational-Image-Recognition-Chatbot

## Overview
The Conversational Image Recognition Chatbot is an AI-powered application that understands images and generates natural language responses. The system uses the BLIP (Bootstrapping Language-Image Pretraining) model to analyze images and generate captions or answers based on the image content. Users can upload an image and interact with the chatbot to get descriptions or information about the image.

##  Technologies Used

- Python
- BLIP (Bootstrapping Language-Image Pretraining)
- Tensorflow
- Hugging Face Transformers
- OpenCV
- FastAPI
  
## Installation section
1.Install required dependencies
pip install torch torchvision transformers fastapi uvicorn pillow python-multipart

2.Run the application
uvicorn app:app --reload

## Usage

- Upload an image through the chatbot interface.
- The system analyzes the image using the trained model.
- The chatbot provides information about the detected object.
