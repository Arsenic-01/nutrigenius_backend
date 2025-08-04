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
from ddgs import DDGS
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.query import Query
from appwrite.exception import AppwriteException
from dotenv import load_dotenv


# ---------- Environment Variables ----------
load_dotenv()


# ---------- Appwrite Setup ----------
appwrite_client = Client()
appwrite_client.set_endpoint(os.environ["APPWRITE_ENDPOINT"])
appwrite_client.set_project(os.environ["APPWRITE_PROJECT_ID"])
appwrite_client.set_key(os.environ["APPWRITE_API_KEY"])

databases = Databases(appwrite_client)
DATABASE_ID = os.environ["APPWRITE_DATABASE_ID"]
COLLECTION_ID = os.environ["APPWRITE_COLLECTION_ID"]

print("✅ Appwrite client initialized.")

# ---------- FastAPI Setup ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

class SaveRequest(BaseModel):
    user_id: str  # Clerk user ID
    recipe_id: int

class UnsaveRequest(BaseModel):
    user_id: str
    recipe_id: int


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

@app.post("/save-recipe")
async def save_recipe(data: SaveRequest):
    """
    Saves a recipe. Relies on a database-level unique index on (userId, recipeId)
    to prevent duplicates. This is the most efficient and reliable method.
    """
    try:
        # This single call attempts to create the document.
        # If the (userId, recipeId) pair already exists, Appwrite will
        # reject this request and the `except` block below will execute.
        response = databases.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            document_id="unique()",
            data={
                "userId": data.user_id,
                "recipeId": data.recipe_id
            }
        )
        # This part only runs if the creation was successful (i.e., no duplicate was found).
        return {"message": "Recipe saved successfully!", "document_id": response['$id']}

    except AppwriteException as e:
        # The Appwrite SDK converts HTTP status codes into error codes.
        # A 409 Conflict error from the server is caught here.
        if e.code == 409:
            # This block runs ONLY when the unique index constraint is violated.
            print(f"Duplicate save attempt by User {data.user_id} for Recipe {data.recipe_id}.")
            return {"message": "Recipe is already in your saved list."}
        
        # If the error was something else (e.g., server down, permissions error),
        # it will be caught here and raised as a 500 error.
        print(f"[ERROR] An unexpected Appwrite error occurred while saving recipe: {e}")
        raise HTTPException(status_code=500, detail="Could not save recipe due to a server error.")


@app.get("/saved-recipes/{user_id}", response_model=List[RecipeResponse])
async def get_saved_recipes(user_id: str):
    """
    Fetches the full recipe data for all recipes a user has saved.
    Simplified version that assumes no duplicate entries exist.
    """
    try:
        response = databases.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            queries=[Query.equal("userId", user_id), Query.limit(500)]
        )

        saved_ids = [doc['recipeId'] for doc in response['documents']]

        if not saved_ids:
            return []

        saved_recipes_df = model_df.loc[model_df.index.isin(saved_ids)]

        if saved_recipes_df.empty:
            return []
        
        # The rest of the image fetching logic remains the same
        recipes_to_process = []
        image_tasks = []
        recipes_needing_image = []

        for idx, row in saved_recipes_df.iterrows():
            if idx not in saved_ids:
                continue
            base_recipe = { "id": idx, **row.to_dict() }
            existing_image_url = row.get("image-url")
            if pd.notna(existing_image_url) and "http" in str(existing_image_url):
                base_recipe["image"] = existing_image_url
                recipes_to_process.append(base_recipe)
            else:
                image_tasks.append(fetch_duckduckgo_image_url(row["RecipeName"]))
                recipes_needing_image.append(base_recipe)
        
        if image_tasks:
            new_image_urls = await asyncio.gather(*image_tasks)
            for recipe_info, new_url in zip(recipes_needing_image, new_image_urls):
                recipe_info["image"] = new_url
                if 'image-url' not in model_df.columns:
                    model_df['image-url'] = pd.Series(dtype='object')
                model_df.loc[recipe_info["id"], "image-url"] = new_url
            recipes_to_process.extend(recipes_needing_image)
        
        recipe_map = {recipe['id']: recipe for recipe in recipes_to_process}
        # Build the final list in the original order
        final_recipes_data = [recipe_map[rid] for rid in saved_ids if rid in recipe_map]
        return [RecipeResponse(**recipe) for recipe in final_recipes_data]

    except AppwriteException as e:
        print(f"[ERROR] Failed to fetch saved recipes for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch saved recipes")

@app.post("/unsave-recipe")
async def unsave_recipe(data: UnsaveRequest):
    """
    Finds and deletes a saved recipe document for a specific user and recipe.
    """
    try:
        # 1. Find the document that matches both the user and recipe ID.
        documents = databases.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            queries=[
                Query.equal("userId", data.user_id),
                Query.equal("recipeId", data.recipe_id),
                Query.limit(1) # We only need one document to get its ID
            ]
        )

        # 2. Check if a document was found.
        if documents['total'] == 0:
            # This can happen in rare cases, but it's good to handle.
            # We don't raise an error, we just confirm it's not there.
            print(f"No saved recipe found for user {data.user_id} and recipe {data.recipe_id} to unsave.")
            return {"message": "Recipe was not saved."}

        # 3. Get the unique document ID ($id) of the found document.
        document_id_to_delete = documents['documents'][0]['$id']

        # 4. Delete that specific document.
        databases.delete_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            document_id=document_id_to_delete
        )

        return {"message": "Recipe unsaved successfully."}

    except AppwriteException as e:
        print(f"[ERROR] Failed to unsave recipe for user {data.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to unsave recipe.")





