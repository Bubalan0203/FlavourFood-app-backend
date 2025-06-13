import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import numpy as np
import os
import shutil
import clip
import torch

model, preprocess = clip.load("ViT-B/32")
print("Model loaded successfully!")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


OPENROUTER_API_KEY = "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"
YOUTUBE_API_KEY = "AIzaSyAX5n6yuCfFKPdIeD9bOsL8eBc8SYsjUGg"  # 🔐 Replace this

class IngredientRequest(BaseModel):
    ingredients: str
    food_type: str

class RecipeDetailRequest(BaseModel):
    title: str
    youtube: str


def fetch_youtube_video_link(query):
    try:
        yt_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=1&q={query}&key={YOUTUBE_API_KEY}"
        response = requests.get(yt_url)
        data = response.json()
        if data.get("items"):
            video = data["items"][0]
            if video["id"]["kind"] == "youtube#video":
                video_id = video["id"]["videoId"]
                return f"https://www.youtube.com/watch?v={video_id}"
    except Exception as e:
        print("YouTube Fetch Error:", e)
    return None

@app.post("/api/recipes")
async def get_recipes(req: IngredientRequest):
    prompt = f"""
Suggest 3 {req.food_type} recipes a college student can cook using: {req.ingredients}.
Only return the recipe titles in JSON format:
["Dish One", "Dish Two", "Dish Three"]
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{ "role": "user", "content": prompt }]
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)
    try:
        recipe_titles = json.loads(res.json()["choices"][0]["message"]["content"])
        results = []
        for idx, title in enumerate(recipe_titles):
            youtube_link = fetch_youtube_video_link(title)
            if youtube_link:
                results.append({"id": str(idx+1), "title": title, "youtube": youtube_link})
        return results
    except Exception as e:
        return {"error": "Failed to parse or enrich recipes", "details": str(e)}



@app.post("/api/generate-steps", response_class=PlainTextResponse)
async def generate_steps(data: RecipeDetailRequest):
    prompt = f"""
You are a cooking assistant. A user is watching a YouTube video titled "{data.title}" here: {data.youtube}.

Based on the title and video context, generate a simple step-by-step recipe. Keep it practical and concise.

Respond only in this format:
1. Step one...
2. Step two...
...
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{ "role": "user", "content": prompt }]
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)
    return res.json()["choices"][0]["message"]["content"]

@app.post("/api/review-dish-image")
async def review_dish_image(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(status_code=400, content={"feedback": "No file uploaded"})

    try:
        # Save uploaded image temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and preprocess image
        image = Image.open(temp_path).convert("RGB")
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        image_input = preprocess(image).unsqueeze(0)

        # Define quality-related descriptions
        descriptions = [
            "a well-plated delicious dish",
            "a dish with burnt edges",
            "a poorly presented meal",
            "a beautifully garnished plate",
            "a messy food presentation"
        ]
        tokens = clip.tokenize(descriptions)

        # Run model
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(tokens)

            logits_per_image, _ = model(image_input, tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Interpret result
        best_match_index = int(np.argmax(probs))
        confidence = probs[0][best_match_index]
        description = descriptions[best_match_index]

        # Custom feedback
        feedback_map = {
            "a well-plated delicious dish": "🤩 That looks delicious and professionally plated!",
            "a dish with burnt edges": "⚠️ Seems slightly overcooked. Try monitoring the heat closely.",
            "a poorly presented meal": "🧐 Taste may be great, but let's work on presentation!",
            "a beautifully garnished plate": "🌿 Gorgeous garnishing! Great attention to detail.",
            "a messy food presentation": "🧽 Maybe tidy up the plating a bit for a better look."
        }

        os.remove(temp_path)
        return {
            "feedback": feedback_map[description],
            "confidence": f"{confidence * 100:.2f}%",
            "label": description
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"feedback": "Server error", "error": str(e)})
