import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from model import get_meal_recommendations, df as model_df
import re
from ddgs import DDGS

# ---------- FastAPI Setup ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Data Models ----------
class RecipeRequest(BaseModel):
    height_cm: float
    weight_kg: float
    desired_ingredients: str
    meal_type: str
    weight_goal: str
    user_allergies: Optional[str] = None
    diet_preference: Optional[str] = None
    max_cooking_time: Optional[int] = None

class RecipeResponse(BaseModel):
    id: int
    RecipeName: str
    Cuisine: Optional[str] = None
    Course: Optional[str] = None
    Diet: Optional[str] = None
    URL: Optional[str] = None
    image: str
    Servings: Optional[int] = None
    PrepTimeInMins: Optional[int] = None
    CookTimeInMins: Optional[int] = None
    TotalTimeInMins: Optional[int] = None

class ProcedureResponse(BaseModel):
    steps: List[str]

# ---------- Image Fetch Utility ----------
async def fetch_duckduckgo_image_url(query: str) -> str:
    placeholder_url = "/generate_image/placeholder.png"

    def is_invalid(url: str) -> bool:
        return "archanaskitchen.com" in url

    def search_image():
        with DDGS() as ddgs:
            results = ddgs.images(
                query=f"{query} food recipe",
                max_results=10,
                safesearch="on"
            )
            for result in results:
                image_url = result.get("image")
                if image_url and not is_invalid(image_url):
                    return image_url
        return placeholder_url

    try:
        return await asyncio.to_thread(search_image)
    except Exception as e:
        print(f"[ERROR] Failed to fetch image for '{query}': {e}")
        return placeholder_url

# ---------- API Routes ----------
@app.get("/")
def root():
    return {"message": "NutriGenius API is running"}

@app.post("/recommend", response_model=List[RecipeResponse])
async def recommend(data: RecipeRequest):
    recommendations_df = get_meal_recommendations(
        height_cm=data.height_cm,
        weight_kg=data.weight_kg,
        meal_type=data.meal_type,
        weight_goal=data.weight_goal,
        desired_ingredients=data.desired_ingredients,
        user_allergies=data.user_allergies or "none",
        max_cooking_time=data.max_cooking_time,
        diet_preference=data.diet_preference,
        num_results=12
    )

    if recommendations_df.empty:
        return []

    recipes = []
    image_tasks = []

    for idx, row in recommendations_df.iterrows():
        base = {
            "id": idx,
            "RecipeName": row["RecipeName"],
            "Cuisine": row.get("Cuisine"),
            "Course": row.get("Course"),
            "Diet": row.get("Diet"),
            "URL": row.get("URL"),
            "Servings": row.get("Servings"),
            "PrepTimeInMins": row.get("PrepTimeInMins"),
            "CookTimeInMins": row.get("CookTimeInMins"),
            "TotalTimeInMins": row.get("TotalTimeInMins")
        }
        recipes.append(base)
        image_tasks.append(fetch_duckduckgo_image_url(row["RecipeName"]))

    image_urls = await asyncio.gather(*image_tasks)

    result = []
    for recipe, image_url in zip(recipes, image_urls):
        recipe["image"] = image_url
        result.append(RecipeResponse(**recipe))

    return result

@app.get("/procedure/{recipe_id}", response_model=ProcedureResponse)
def get_procedure(recipe_id: int):
    try:
        row = model_df.loc[recipe_id]
        instructions = row.get("Instructions", "")
        if not isinstance(instructions, str) or not instructions.strip():
            raise HTTPException(status_code=404, detail="No procedure found for this recipe")
        steps = [step.strip() for step in re.split(r"\.|\n", instructions) if step.strip()]
        return {"steps": steps}
    except KeyError:
        raise HTTPException(status_code=404, detail="Recipe not found")
    except Exception as e:
        print(f"[ERROR] Failed to get procedure for ID {recipe_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
