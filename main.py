import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
import io
import clip
import json

# 🚀 FastAPI app setup
app = FastAPI()

# 🌐 Allow all CORS origins (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 🔐 API KEYS
OPENROUTER_API_KEY = "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"
YOUTUBE_API_KEY = "AIzaSyAX5n6yuCfFKPdIeD9bOsL8eBc8SYsjUGg"

# 🥘 Request Models
class IngredientRequest(BaseModel):
    ingredients: str
    food_type: str

class RecipeDetailRequest(BaseModel):
    title: str
    youtube: str

# 🔎 YouTube Video Link Helper
def fetch_youtube_video_link(query):
    try:
        yt_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=1&q={query}&key={YOUTUBE_API_KEY}"
        response = requests.get(yt_url)
        data = response.json()
        if data.get("items"):
            video_id = data["items"][0]["id"]["videoId"]
            return f"https://www.youtube.com/watch?v={video_id}"
    except Exception as e:
        print("YouTube Fetch Error:", e)
    return None

# 🍛 Get Recipe Titles
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
            results.append({"id": str(idx+1), "title": title, "youtube": youtube_link})
        return results
    except Exception as e:
        return {"error": "Failed to parse or enrich recipes", "details": str(e)}

# 📋 Generate Recipe Steps
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

# 🤖 Load CLIP model
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 🍽️ Labels to detect
labels = [
    "burnt", "overcooked", "undercooked", "garnished", "neatly plated",
    "spilled", "crispy", "raw", "golden brown", "dry", "soggy", "colorful", "messy"
]

def analyze_with_clip(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(label) for label in labels]).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        top_indices = probs.argsort()[-3:][::-1]
        top_labels = [labels[i] for i in top_indices if probs[i] > 0.1]
        top_confidence = probs[top_indices[0]]  # highest prob

        return top_labels, top_confidence
    except Exception as e:
        print("CLIP Error:", e)
        return [], 0

def get_feedback_from_openrouter(concepts):
    prompt = (
        f"You are a professional chef and food reviewer. "
        f"Based on these visual cues: {', '.join(concepts)}, "
        f"give a short feedback in **exactly two sentences**. "
        f"First sentence should be a compliment (if any), second should be a polite improvement suggestion (if needed)."
    )

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("OpenRouter Error:", e)
        return "❌ Failed to analyze the image. Please try again."
@app.post("/api/review-dish-image")
async def review_dish_image(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(status_code=400, content={"feedback": "No file uploaded"})

    try:
        image_bytes = await file.read()
        concepts, confidence = analyze_with_clip(image_bytes)

        food_keywords = ["garnished", "crispy", "burnt", "overcooked", "plated", "golden brown", "soggy", "dry", "colorful"]

        is_food = any(label in concepts for label in food_keywords)

        # Reject if not food or low confidence
        if not is_food or confidence < 0.1:
            return JSONResponse(
                status_code=200,
                content={"concepts": [], "feedback": "🚫 This doesn't appear to be a valid food image. Please try again with a dish photo."}
            )

        feedback = get_feedback_from_openrouter(concepts)
        return {"concepts": concepts, "feedback": feedback}

    except Exception as e:
        print("Server Error:", e)
        return JSONResponse(status_code=500, content={"feedback": "Server error", "error": str(e)})
