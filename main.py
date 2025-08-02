import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from model import get_meal_recommendations
import model
import re
# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Google API keys for image search
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
    print("⚠️ WARNING: GOOGLE_API_KEY or SEARCH_ENGINE_ID is missing from your .env file")

# ---------------------- Utility Functions ----------------------

async def fetch_google_image_url(query: str, client: httpx.AsyncClient) -> str:
    """
    Fetches the first image result URL from Google Custom Search API.
    """
    if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
        return "/generate_image/placeholder.png"
        
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": f"{query}",
        "searchType": "image",
        "num": 1,
        "imgSize": "large"
    }
    try:
        response = await client.get(search_url, params=params)
        response.raise_for_status()
        results = response.json()
        if "items" in results and len(results["items"]) > 0:
            return results["items"][0]["link"]
    except Exception as e:
        print(f"[ERROR] Failed image fetch for '{query}': {e}")
    
    return "/generate_image/placeholder.png"

# ---------------------- Mock Procedures ----------------------

mock_procedures = {
    1: [
        "Heat butter and oil in a pan; sauté whole spices until fragrant.",
        "Add finely chopped onions and cook until golden brown.",
        "Stir in ginger-garlic paste, then add tomato puree and cashew paste.",
        "Add turmeric, red chili powder, and garam masala. Cook until oil separates.",
        "Blend the gravy into a smooth, creamy texture.",
        "Add paneer cubes, a splash of cream, and kasuri methi. Simmer for 5 minutes.",
        "Garnish with fresh coriander and serve hot with naan or rice."
    ],
    2: [
        "Wash and soak basmati rice for 20 minutes. Drain completely.",
        "Heat ghee or oil in a pressure cooker. Add cumin seeds, bay leaf, and other whole spices.",
        "Sauté sliced onions until they turn light brown.",
        "Add mixed vegetables like carrots, peas, and beans. Sauté for 2-3 minutes.",
        "Add the soaked rice and stir gently to coat with ghee.",
        "Pour in water and add salt to taste. Stir well.",
        "Pressure cook for 2 whistles. Let the pressure release naturally before opening.",
        "Fluff the rice with a fork and serve hot with raita."
    ],
    3: [
        "Prepare the potato filling: Heat oil, splutter mustard seeds, add onions, and curry leaves.",
        "Add turmeric powder and boiled, mashed potatoes. Mix well and set aside.",
        "Heat a non-stick tawa or griddle. Once hot, wipe it with a damp cloth.",
        "Pour a ladleful of dosa batter in the center and spread it outwards in a circular motion.",
        "Drizzle some oil or ghee around the edges and cook until the dosa is golden and crisp.",
        "Place a portion of the potato filling in the center of the dosa.",
        "Fold the dosa in half and serve immediately with coconut chutney and sambar."
    ]
}

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
    return {"message": "NutriGenius Backend is running!"}


@app.post("/recommend")
async def recommend(data: RecipeRequest):
    """
    Recommend meals based on user data using the ML model.
    """

    # 1️⃣ Get recommendations from the model
    recommendations_df = get_meal_recommendations(
        height_cm=data.height,
        weight_kg=data.weight,
        meal_type=data.mealType,
        weight_goal=data.weightGoal,
        desired_ingredients=data.requiredIngredient,
        user_allergies=data.allergicIngredient or "none"
    )
    
    # 2️⃣ Convert DataFrame → list of dicts
    recipes = []
    for idx, row in recommendations_df.iterrows():
        ingredients_list = [i.strip() for i in row['Ingredients'].split(',')]
        recipes.append({
            "id": idx + 1,
            "title": row['Recipe_Name'],
            "description": f"A delicious {data.mealType} option",
            "ingredients": ingredients_list,
        })

    # 3️⃣ Fetch images for each recipe
    async with httpx.AsyncClient() as client:
        image_tasks = [fetch_google_image_url(recipe["title"], client) for recipe in recipes]
        fetched_urls = await asyncio.gather(*image_tasks)

    # 4️⃣ Attach images to recipes
    for recipe, url in zip(recipes, fetched_urls):
        recipe["image"] = url

    # 5️⃣ Return recipes with images
    return {"recipes": recipes}


@app.get("/procedure/{recipe_id}", response_model=ProcedureResponse)
async def get_procedure(recipe_id: int):
    """
    Return step-by-step procedure for a recipe from the dataset.
    """
    try:
        # Use the recipe_id as DataFrame index
        recipe_row = model.df.iloc[recipe_id]
        instructions = recipe_row.get("TranslatedInstructions", "")

        if not isinstance(instructions, str) or not instructions.strip():
            raise HTTPException(status_code=404, detail="No procedure found for this recipe")

        # Split instructions into step-by-step list
        steps = [step.strip() for step in re.split(r'[.\n]', instructions) if step.strip()]

        return {"steps": steps}

    except IndexError:
        raise HTTPException(status_code=404, detail="Recipe ID not found")
