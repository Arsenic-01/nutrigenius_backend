import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
    print("⚠️ WARNING: GOOGLE_API_KEY or SEARCH_ENGINE_ID is missing from your .env file")


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

class RecipeRequest(BaseModel):
    height: float
    weight: float
    requiredIngredient: str
    allergicIngredient: Optional[str] = None
    estimatedTime: int
    numResults: int

class ProcedureResponse(BaseModel):
    steps: List[str]


@app.get("/")
async def root():
    return {"message": "NutriGenius Backend is running!"}

@app.post("/recommend")
async def recommend(data: RecipeRequest):
    base_recipes = [
        {
            "id": 1,
            "title": "Paneer Butter Masala",
            "description": "A creamy North Indian Curry made with paneer, butter and rich tomato gravy",
            "ingredients": ["Paneer", "Butter", "Masala", "Green Peas", "Curry Leaves", "Coriander"]
        },
        {
            "id": 2,
            "title": "Veg Pulao",
            "description": "Fragrant basmati rice cooked with seasonal vegetables and aromatic spices.",
            "ingredients": ["Rice", "Carrot", "Peas", "Beans", "Ghee", "Spices"]
        },
        {
            "id": 3,
            "title": "Masala Dosa",
            "description": "South Indian delicacy with crispy dosa and spiced potato filling.",
            "ingredients": ["Rice Batter", "Potato", "Onion", "Curry Leaves", "Mustard Seeds", "Chutney"]
        }
    ]

    async with httpx.AsyncClient() as client:
        image_tasks = [fetch_google_image_url(recipe["title"], client) for recipe in base_recipes]
        fetched_urls = await asyncio.gather(*image_tasks)

    final_recipes = []
    for recipe, url in zip(base_recipes, fetched_urls):
        final_recipes.append({**recipe, "image": url})

    return {"recipes": final_recipes}

@app.get("/procedure/{recipe_id}", response_model=ProcedureResponse)
async def get_procedure(recipe_id: int):
    procedure = mock_procedures.get(recipe_id)
    if not procedure:
        raise HTTPException(status_code=404, detail="Recipe procedure not found.")
    return {"steps": procedure}