from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # allow GET, POST, OPTIONS
    allow_headers=["*"],  # allow all headers
)

class RecipeRequest(BaseModel):
    height: float
    weight: float
    requiredIngredient: str
    estimatedTime: int
    numResults: int
    
@app.get("/")
async def root():
    return {"message": "NutriGenius Backend is running!"}

@app.post("/recommend")
async def recommend(data: RecipeRequest):
    print("Received data: ", data.model_dump())
    
    dummy_recipes = [
        {
            "title": "Paneer Butter Masala",
            "description": "A creamy North Indian Curry made with paneer, butter andf rich tomato gravy",
            "image": "/recipes/PaneerButterMasala.jpeg",
            "ingredients": ["Paneer", "Butter", "Masala", "Green Peas", "Curry Leaves", "Coriander"]
        },
        {
            "title": "Veg Pulao",
            "description": "Fragrant basmati rice cooked with seasonal vegetables and aromatic spices.",
            "image": "/recipes/VegPulaoo.jpg",
            "ingredients": ["Rice", "Carrot", "Peas", "Beans", "Ghee", "Spices"]
        },
        {
            "title": "Masala Dosa",
            "description": "South Indian delicacy with crispy dosa and spiced potato filling.",
            "image": "/recipes/MasalaDosa.jpg",
            "ingredients": ["Rice Batter", "Potato", "Onion", "Curry Leaves", "Mustard Seeds", "Chutney"]
        }
    ]
    
    return{
        "recipes": dummy_recipes
    }