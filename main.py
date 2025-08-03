import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from model import get_meal_recommendations, df as model_df
import re
import math
import pandas as pd

# Use the correct, modern ddgs library
# Make sure you have run: pip install -U ddgs
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

# --- Add a shutdown event to save the updated dataset ---
@app.on_event("shutdown")
def shutdown_event():
    """
    This function is called when the server is shutting down.
    It saves the dataframe with the new image URLs back to the CSV.
    """
    print("Server shutting down, saving updated dataset...")
    try:
        # Ensure the 'image-url' column exists before saving
        if 'image-url' not in model_df.columns:
            model_df['image-url'] = None
        model_df.to_csv('master_recipe_dataset.csv', index=False)
        print("✅ Dataset saved successfully.")
    except Exception as e:
        print(f"❌ ERROR saving dataset: {e}")


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
    skill_level: Optional[str] = None
    pantry_ingredients: Optional[str] = None
    cuisine_preference: Optional[List[str]] = None
    page: int = 1

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

class PaginatedRecipeResponse(BaseModel):
    recipes: List[RecipeResponse]
    page: int
    total_pages: int
    has_next_page: bool

class ProcedureResponse(BaseModel):
    steps: List[str]

# ---------- Image Fetch Utility ----------
async def fetch_duckduckgo_image_url(query: str) -> str:
    """
    Asynchronously fetches the first valid image result URL from DuckDuckGo using ddgs.
    """
    placeholder_url = "https://placehold.co/600x400/EEE/31343C?text=Image+Not+Available"

    try:
        # --- FIX: Use asyncio.to_thread to run the synchronous ddgs library correctly ---
        def sync_duckduckgo_search():
            # The 'ddgs' library expects the parameter to be 'keywords'
            with DDGS() as ddgs:
                results = ddgs.images(query=f"{query} food recipe", max_results=5)
                if not results:
                    return placeholder_url
                for result in results:
                    image_url = result.get("image")
                    if image_url and "archanaskitchen.com" not in image_url:
                        return image_url
            return placeholder_url
        
        return await asyncio.to_thread(sync_duckduckgo_search)

    except Exception as e:
        print(f"[ERROR] Failed to fetch image for '{query}': {e}")
        return placeholder_url

# ---------- API Routes ----------
@app.get("/")
async def root():
    return {"message": "NutriGenius API is running"}

@app.post("/recommend", response_model=PaginatedRecipeResponse)
async def recommend(data: RecipeRequest):
    """
    Recommend meals with pagination and intelligent image caching.
    """
    PAGE_SIZE = 9

    loop = asyncio.get_event_loop()
    model_args = {
        "height_cm": data.height_cm,
        "weight_kg": data.weight_kg,
        "desired_ingredients": data.desired_ingredients,
        "user_allergies": data.user_allergies or "none",
        "meal_type": data.meal_type,
        "weight_goal": data.weight_goal,
        "max_cooking_time": data.max_cooking_time,
        "diet_preference": data.diet_preference,
        "cuisine_preference": data.cuisine_preference,
        "skill_level": data.skill_level,
        "pantry_ingredients": data.pantry_ingredients,
        "num_results": 100
    }
    recommendations_df = await loop.run_in_executor(
        None,
        lambda: get_meal_recommendations(**model_args)
    )

    if recommendations_df.empty:
        return {"recipes": [], "page": 1, "total_pages": 0, "has_next_page": False}

    total_results = len(recommendations_df)
    total_pages = math.ceil(total_results / PAGE_SIZE)
    start_index = (data.page - 1) * PAGE_SIZE
    end_index = start_index + PAGE_SIZE
    
    paginated_df = recommendations_df.iloc[start_index:end_index]

    if paginated_df.empty:
        return {"recipes": [], "page": data.page, "total_pages": total_pages, "has_next_page": False}

    # --- Intelligent Image Fetching Logic ---
    recipes_to_process = []
    image_tasks = []
    recipes_needing_image = []

    for idx, row in paginated_df.iterrows():
        base_recipe = { "id": idx, **row.to_dict() }
        
        # Check if a valid image URL already exists in the dataset
        existing_image_url = row.get("image-url")
        if pd.notna(existing_image_url) and "http" in str(existing_image_url):
            base_recipe["image"] = existing_image_url
            recipes_to_process.append(base_recipe)
        else:
            # If not, add a task to fetch it and mark it for update
            image_tasks.append(fetch_duckduckgo_image_url(row["RecipeName"]))
            recipes_needing_image.append(base_recipe)
    
    # Fetch all missing images concurrently
    if image_tasks:
        print(f"Fetching {len(image_tasks)} new images...")
        new_image_urls = await asyncio.gather(*image_tasks)
        
        # Update the recipes and the main dataframe with the new URLs
        for recipe_info, new_url in zip(recipes_needing_image, new_image_urls):
            recipe_info["image"] = new_url
            # Update the main dataframe in memory
            # Ensure the column exists before trying to write to it
            if 'image-url' not in model_df.columns:
                model_df['image-url'] = None
            model_df.loc[recipe_info["id"], "image-url"] = new_url
        
        # Add the newly processed recipes to the main list
        recipes_to_process.extend(recipes_needing_image)
        # Re-sort to maintain original recommendation order
        recipes_to_process.sort(key=lambda x: paginated_df.index.get_loc(x['id']))


    final_recipes = [RecipeResponse(**recipe) for recipe in recipes_to_process]

    return {
        "recipes": final_recipes,
        "page": data.page,
        "total_pages": total_pages,
        "has_next_page": data.page < total_pages
    }

@app.get("/procedure/{recipe_id}", response_model=ProcedureResponse)
async def get_procedure(recipe_id: int):
    try:
        row = model_df.loc[recipe_id]
        instructions = row.get("Instructions", "")
        if not isinstance(instructions, str) or not instructions.strip():
            raise HTTPException(status_code=404, detail="No procedure found")
        steps = [step.strip() for step in re.split(r"\.|\n", instructions) if step.strip()]
        return {"steps": steps}
    except KeyError:
        raise HTTPException(status_code=404, detail="Recipe not found")
    except Exception as e:
        print(f"[ERROR] Failed to get procedure for ID {recipe_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
