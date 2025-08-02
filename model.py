import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# --- 1. Data Loading and Initial Preparation ---

# Load the dataset from the provided Google Sheet URL
google_sheet_url = 'https://docs.google.com/spreadsheets/d/1P0q4lSw-_wSckjWyCyFBvgcKakaeP8OAyaPhGZFoxMU/export?format=csv'
try:
    df = pd.read_csv(google_sheet_url)
    # Clean up column names to remove leading/trailing spaces and special characters
    df.columns = df.columns.str.strip().str.replace('-', '_')
    print("Columns loaded and cleaned:", df.columns.to_list())

except Exception as e:
    print(f"Error loading or processing the dataset: {e}")
    # Fallback dummy data if URL fails
    data = {'TranslatedRecipeName': ['Butter Toast', 'Paneer Salad', 'Veg Curry'],
            'Cleaned_Ingredients': ['bread, butter', 'Paneer, Cabage, rice, broccoli', 'Cauliflower, Potato, asparagus'],
            'TotalTimeInMins': [10, 25, 30]}
    df = pd.DataFrame(data)


# --- 2. Feature Engineering & Preprocessing ---

# Function to derive Meal_Type from recipe name (since the column is missing)
def get_meal_type(recipe_name):
    if not isinstance(recipe_name, str):
        return 'Unknown'
    name = recipe_name.lower()
    if any(keyword in name for keyword in ['egg', 'oat', 'pancake', 'smoothie', 'breakfast']):
        return 'Breakfast'
    if any(keyword in name for keyword in ['sandwich', 'salad', 'wrap', 'lunch']):
        return 'Lunch'
    if any(keyword in name for keyword in ['dinner', 'curry', 'roast', 'steak', 'karela']):
        return 'Dinner'
    return 'Lunch' # Default to Lunch if no keywords match

# Function to preprocess the text data
def preprocess_text(text):
    if isinstance(text, str):
        # Cleans text by removing parentheses and replacing commas with spaces
        return text.lower().replace(',', ' ').replace('(', ' ').replace(')', ' ')
    return ''

# Use a try-except block to handle potential KeyErrors gracefully
try:
    # --- IMPROVEMENT: Use the 'Cleaned_Ingredients' column for better accuracy ---
    RECIPE_NAME_COL = 'TranslatedRecipeName'
    INGREDIENTS_COL = 'Cleaned_Ingredients'
    
    # Create the 'Meal_Type' column
    df['Meal_Type'] = df[RECIPE_NAME_COL].apply(get_meal_type)

    # Process the ingredients text from the cleaned column
    df['Processed_Ingredients'] = df[INGREDIENTS_COL].apply(preprocess_text)

    # Combine ingredients and meal type for the main feature set
    df['Combined_Features'] = df['Processed_Ingredients'] + ' ' + df['Meal_Type'].str.lower()

    # Use a single vectorizer for both ingredients and allergies
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the features
    feature_matrix = vectorizer.fit_transform(df['Combined_Features'])

except KeyError as e:
    print(f"\n---FATAL ERROR---")
    print(f"A required column '{e}' was not found in the dataset.")
    print(f"Please check the 'Columns loaded and cleaned' printout above and ensure the column names in the code match the file exactly.")
    exit()


# --- 3. The Recommendation Function ---

def get_meal_recommendations(height_cm, weight_kg, meal_type, weight_goal, desired_ingredients, user_allergies):
    """
    Generates meal recommendations based on user inputs.
    """
    print("--- Starting Recommendation Process ---")
    
    # --- Create a Feature Vector for the User's Query ---
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    print(f"User BMI calculated: {bmi:.2f} (Goal: {weight_goal})")

    processed_desired_ingredients = preprocess_text(desired_ingredients)
    user_query_text = f"{processed_desired_ingredients} {meal_type.lower()}"
    user_vector = vectorizer.transform([user_query_text])

    # --- Calculate Similarity ---
    ingredient_similarity = cosine_similarity(user_vector, feature_matrix)

    # --- Handle Allergies ---
    allergy_penalty = np.zeros(len(df))
    processed_user_allergies = preprocess_text(user_allergies)
    if processed_user_allergies.strip():
        # We check if any of the user's allergy words appear in the recipe's ingredients
        for i, ingredients in enumerate(df['Processed_Ingredients']):
            # Check for whole word matches to avoid partial matches (e.g., 'rice' in 'price')
            if any(f' {allergen} ' in f' {ingredients} ' for allergen in processed_user_allergies.split()):
                allergy_penalty[i] = 100 # Apply a heavy penalty

    # --- Combine Scores and Get Final Recommendations ---
    meal_type_mask = (df['Meal_Type'].str.lower() == meal_type.lower())
    final_scores = ingredient_similarity.flatten() - allergy_penalty
    final_scores[~meal_type_mask] = -np.inf

    num_recommendations = 5
    valid_indices = np.where(meal_type_mask)[0]
    if len(valid_indices) == 0:
        print(f"No recipes found for the meal type: {meal_type}")
        return pd.DataFrame()
        
    num_recommendations = min(num_recommendations, len(valid_indices))
    
    # Get indices from the filtered list to ensure we only recommend from the correct meal type
    top_indices_in_filtered = np.argsort(final_scores[valid_indices])[-num_recommendations:][::-1]
    recommended_indices = valid_indices[top_indices_in_filtered]

    # Create the output DataFrame using the correct column names
    recommendations_df = pd.DataFrame({
        'Recipe_Name': df.iloc[recommended_indices][RECIPE_NAME_COL],
        'Similarity_Score': final_scores[recommended_indices],
        'Derived_Meal_Type': df.iloc[recommended_indices]['Meal_Type'],
        'Ingredients': df.iloc[recommended_indices][INGREDIENTS_COL]
    })

    print("--- Recommendations Generated ---")
    return recommendations_df


# --- 4. Example Usage (Testing the Model) ---

print("\n" + "="*50)
print("Example 1: Lunch to Lose Weight, wants Chicken, no allergies")
print("="*50)
recommendations = get_meal_recommendations(
    height_cm=175,
    weight_kg=80,
    meal_type='Lunch',
    weight_goal='Lose',
    desired_ingredients='chicken, salad, olive oil',
    user_allergies='none'
)
print(recommendations)
print("\n" * 2)


print("="*50)
print("Example 2: Breakfast to Gain Weight, wants Eggs, allergic to nuts")
print("="*50)
recommendations = get_meal_recommendations(
    height_cm=180,
    weight_kg=70,
    meal_type='Breakfast',
    weight_goal='Gain',
    desired_ingredients='eggs, bacon, toast',
    user_allergies='nuts, almonds, peanut'
)
print(recommendations)
print("\n" * 2)

print("="*50)
print("Example 3: Dinner, wants a vegetarian meal, allergic to shellfish")
print("="*50)
recommendations = get_meal_recommendations(
    height_cm=160,
    weight_kg=60,
    meal_type='Dinner',
    weight_goal='Maintain',
    desired_ingredients='quinoa, beans, vegetables, tomato',
    user_allergies='shellfish, shrimp, crab'
)
print(recommendations)

print("="*50)
print("Example 3: Dinner, wants a vegetarian meal, allergic to shellfish")
print("="*50)
recommendations = get_meal_recommendations(
    height_cm=160,
    weight_kg=60,
    meal_type='Dinner',
    weight_goal='Maintain',
    desired_ingredients='Paneer',
    user_allergies='shellfish, shrimp, crab'
)
print(recommendations)
