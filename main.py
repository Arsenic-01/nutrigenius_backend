import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import model
from model import get_meal_recommendations
import re
import random

# ---------------------- Load Environment Variables ----------------------
load_dotenv()

# Pexels API Key
PEXELS_API_KEY = "mDepqbGjWrDLWhrs5Bymo59epxV6lhLRTcGGOHYF3W9nQ7yfwa3iKD3n"

# ---------------------- FastAPI App Setup ----------------------
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Utility Functions ----------------------

async def fetch_pexels_image_url(
    recipe_title: str,
    ingredients: List[str],
    meal_type: str,
    client: httpx.AsyncClient
) -> str:
    """
    Fetch a context-aware food image from Pexels API for the recipe.
    Priority: recipe title + top ingredient + meal type
    """
    headers = {"Authorization": PEXELS_API_KEY}
    url = "https://api.pexels.com/v1/search"

    # Create multiple queries to improve image relevance
    main_ingredient = ingredients[0] if ingredients else ""
    queries = [
        f"{recipe_title} {meal_type} food",
        f"{main_ingredient} {meal_type} recipe",
        f"{recipe_title} dish",
        f"{main_ingredient} dish",
    ]

    for query in queries:
        params = {"query": query, "per_page": 10}
        try:
            response = await client.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            photos = data.get("photos", [])
            if photos:
                chosen = random.choice(photos)["src"]["medium"]
                print(f"[INFO] Image found for '{query}': {chosen}")
                return chosen
        except Exception as e:
            print(f"[WARN] Pexels fetch failed for query '{query}': {e}")

    # Final fallback placeholder
    print("[WARN] No image found. Using placeholder.")
    return "https://via.placeholder.com/300x200?text=No+Image"

# ---------------------- Request & Response Models ----------------------

class RecipeRequest(BaseModel):
    height: float
    weight: float
    requiredIngredient: str
    allergicIngredient: Optional[str] = None
    mealType: str = "Lunch"
    weightGoal: str = "Maintain"
    estimatedTime: int
    numResults: int

class ProcedureResponse(BaseModel):
    steps: List[str]

# ---------------------- Routes ----------------------

@app.get("/")
async def root():
    return {"message": "NutriGenius Backend is running with Pexels images!"}

@app.post("/recommend")
async def recommend(data: RecipeRequest):
    """
    Recommend meals with randomization and Pexels images.
    """
    # 1️⃣ Get recommendations from ML model
    recommendations_df = get_meal_recommendations(
        height_cm=data.height,
        weight_kg=data.weight,
        meal_type=data.mealType,
        weight_goal=data.weightGoal,
        desired_ingredients=data.requiredIngredient,
        user_allergies=data.allergicIngredient or "none"
    )

    if recommendations_df.empty:
        raise HTTPException(status_code=404, detail="No recipes found for given criteria")

    # 2️⃣ Shuffle & randomly pick recipes
    recommendations_df = recommendations_df.sample(frac=1).head(data.numResults)

    # 3️⃣ Build recipe list
    recipes = []
    for idx, row in recommendations_df.iterrows():
        ingredients_list = [i.strip() for i in row['Ingredients'].split(',')]
        recipes.append({
            "id": int(idx),
            "title": row['Recipe_Name'],
            "description": f"A delicious {data.mealType} option",
            "ingredients": ingredients_list,
        })

    # 4️⃣ Fetch context-aware images asynchronously
    async with httpx.AsyncClient() as client:
        image_tasks = [
            fetch_pexels_image_url(
                recipe["title"],
                recipe["ingredients"],
                data.mealType,
                client
            )
            for recipe in recipes
        ]
        fetched_urls = await asyncio.gather(*image_tasks)

    # 5️⃣ Attach images to recipes
    for recipe, url in zip(recipes, fetched_urls):
        recipe["image"] = url

    return {"recipes": recipes}

@app.get("/procedure/{recipe_id}", response_model=ProcedureResponse)
async def get_procedure(recipe_id: int):
    """
    Return step-by-step procedure for a recipe from the dataset.
    """
    try:
        recipe_row = model.df.iloc[recipe_id]
        instructions = recipe_row.get("TranslatedInstructions", "")

        if not isinstance(instructions, str) or not instructions.strip():
            raise HTTPException(status_code=404, detail="No procedure found for this recipe")

        steps = [step.strip() for step in re.split(r'[.\n]', instructions) if step.strip()]
        return {"steps": steps}

    except IndexError:
        raise HTTPException(status_code=404, detail="Recipe ID not found")
