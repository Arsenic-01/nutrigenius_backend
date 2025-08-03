import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import time

# --- 1. EFFICIENT & FOCUSED Data Loading ---

# Load the pre-compiled master recipe dataset
try:
    start_time = time.time()
    df = pd.read_csv('master_recipe_dataset.csv')
    # Fill any potential missing values in key numeric columns to prevent errors
    numeric_cols = ['Calories (kcal)', 'Protein (g)', 'Fibre (g)', 'rating_avg', 'n_ratings']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    end_time = time.time()
    print(f"Master recipe dataset loaded successfully in {end_time - start_time:.2f} seconds.")
except FileNotFoundError:
    print("FATAL ERROR: 'master_recipe_dataset.csv' not found. Please run the 'create_master_dataset.py' script first.")
    sys.exit()

# --- 2. One-Time Vectorization & Preprocessing ---
start_time = time.time()

# --- NEW: More Granular Meal Type Classification ---
def get_meal_type(course):
    if not isinstance(course, str): return 'Other'
    course = course.lower()
    if 'breakfast' in course: return 'Breakfast'
    if 'main course' in course or 'dinner' in course or 'lunch' in course: return 'Main Course'
    if 'side dish' in course: return 'Side Dish'
    if 'snack' in course or 'appetizer' in course: return 'Snack'
    if 'dessert' in course or 'sweet' in course: return 'Dessert'
    return 'Other'

df['Meal_Type'] = df['Course'].apply(get_meal_type)

df['Combined_Features'] = (df['Cleaned_Ingredients'].fillna('') + ' ' + 
                           df['Diet'].fillna('') + ' ' + 
                           df['Cuisine'].fillna(''))
                           
vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['Combined_Features'])
end_time = time.time()
print(f"TF-IDF feature matrix and feature engineering created in {end_time - start_time:.2f} seconds.")


# --- 3. The Recommendation Function (Now with Course-Aware Ranking) ---

def get_meal_recommendations(
    height_cm, weight_kg, desired_ingredients, user_allergies, meal_type, weight_goal,
    max_cooking_time=None, diet_preference=None, cuisine_preference=None, skill_level=None, pantry_ingredients=None, num_results=100
):
    print("\n--- Starting Advanced Recommendation Process ---")
    
    # --- Stage 1: Fast Filtering ---
    # We still filter by the general meal type from the frontend (e.g., "Dinner")
    # The model will handle the fine-tuning
    if meal_type.lower() in ['lunch', 'dinner']:
        meal_mask = df['Meal_Type'].isin(['Main Course', 'Side Dish', 'Snack'])
    else:
        meal_mask = df['Meal_Type'].str.lower() == meal_type.lower()

    base_candidate_df = df[meal_mask].copy()
    
    # (Other filters remain the same)
    if cuisine_preference:
        base_candidate_df = base_candidate_df[base_candidate_df['Cuisine'].isin(cuisine_preference)]
    if diet_preference:
        base_candidate_df = base_candidate_df[base_candidate_df['Diet'].str.contains(diet_preference, case=False, na=False)]
    if max_cooking_time:
        base_candidate_df = base_candidate_df[base_candidate_df['TotalTimeInMins'] <= max_cooking_time]
    
    candidate_df = base_candidate_df
    # (Strict ingredient filtering remains the same)
    if desired_ingredients and desired_ingredients.lower().strip() != 'none':
        must_have_ingredients = [ing.strip().lower() for ing in desired_ingredients.split(',') if ing.strip()]
        strict_mask = candidate_df['Cleaned_Ingredients'].str.lower().apply(
            lambda x: all(ing in x for ing in must_have_ingredients)
        )
        strict_filtered_df = candidate_df[strict_mask]
        if not strict_filtered_df.empty:
            candidate_df = strict_filtered_df.copy()
        else:
            relaxed_mask = candidate_df['Cleaned_Ingredients'].str.lower().apply(
                lambda x: any(ing in x for ing in must_have_ingredients)
            )
            candidate_df = candidate_df[relaxed_mask].copy()

    if candidate_df.empty: return pd.DataFrame()

    # --- Stage 2: Fast Ranking ---
    user_query_text = f"{desired_ingredients} {diet_preference or ''}"
    user_vector = vectorizer.transform([user_query_text])
    
    candidate_indices = candidate_df.index
    candidate_matrix = feature_matrix[candidate_indices]

    candidate_df.loc[:, 'text_similarity'] = cosine_similarity(user_vector, candidate_matrix).flatten()
    
    bmi = (weight_kg / ((height_cm / 100) ** 2)) if height_cm > 0 else 22

    def calculate_final_score(row):
        weights = {"text": 0.4, "health": 0.3, "course": 0.2, "pantry": 0.1}
        score = row['text_similarity'] * weights["text"]
        
        course_score = 0
        if meal_type.lower() in ['lunch', 'dinner']:
            if row['Meal_Type'] == 'Main Course':
                course_score = 1.0  # High bonus for main courses
            elif row['Meal_Type'] == 'Side Dish':
                course_score = 0.5  # Medium bonus for side dishes
            elif row['Meal_Type'] == 'Snack':
                course_score = -0.5 # Penalty for snacks
        score += course_score * weights["course"]

        health_score = 0
        calories = row.get('Calories (kcal)', 0)
        protein = row.get('Protein (g)', 0)
        fiber = row.get('Fibre (g)', 0)
        if pd.notna(calories):
            if weight_goal == 'Lose' or bmi > 25:
                health_score += max(0, 1 - (calories / 600))
                health_score += (fiber / 15) if pd.notna(fiber) else 0
            elif weight_goal == 'Gain' or bmi < 18.5:
                health_score += min(1, calories / 800)
                health_score += (protein / 40) if pd.notna(protein) else 0
        score += health_score * weights["health"]
        
        if pantry_ingredients:
            pantry_items = [p.strip() for p in pantry_ingredients.lower().split(',')]
            recipe_items = set(row['Cleaned_Ingredients'].lower().split(','))
            matches = sum(1 for item in pantry_items if item in recipe_items)
            score += (matches / max(1, len(pantry_items))) * weights["pantry"]

        return score

    candidate_df.loc[:, 'final_score'] = candidate_df.apply(calculate_final_score, axis=1)
    recommendations = candidate_df.sort_values(by='final_score', ascending=False).head(num_results)
    
    print("--- Recommendations Generated ---")
    return recommendations

# --- 4. Example Usage & Testing ---
if __name__ == '__main__':
    print("\n" + "="*50)
    print("RUNNING SAMPLE TEST CASE")
    print("="*50)

    sample_input = {
        "height_cm": 175,
        "weight_kg": 70,
        "desired_ingredients": "Paneer",
        "user_allergies": "none",
        "meal_type": "Dinner",
        "weight_goal": "Maintain",
        "diet_preference": "Vegetarian",
        "skill_level": "Intermediate",
    }

    recommendations = get_meal_recommendations(**sample_input)

    if not recommendations.empty:
        print("\nTop Recommendations:")
        print(recommendations[[
            'RecipeName', 
            'final_score', 
            'Cuisine', 
            'Diet', 
            'Meal_Type',
            'Calories (kcal)'
        ]].round(2))
    else:
        print("\n--> Final Result: No recommendations found for the sample case.")
