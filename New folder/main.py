import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import shutil
import os

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
).to(device)
model.eval()

def get_answer(image_path, question):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs)

    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer

app = FastAPI()

@app.post("/chat")
async def chat(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    image_path = f"temp_{file.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    answer = get_answer(image_path, question)

    os.remove(image_path)

    return {
        "question": question,
        "answer": answer
    }