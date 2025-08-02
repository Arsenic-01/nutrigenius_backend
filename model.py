import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import sys

# --- 1. Data Loading and Initial Preparation ---

local_dataset = 'NutriGeniusDataset.csv'

try:
    df = pd.read_csv(local_dataset)
    if df.empty:
        raise ValueError("The loaded DataFrame is empty. Check the Google Sheet URL or its content.")

    # Clean column names before using them
    df.columns = df.columns.str.strip()

    # Define required columns for the script to function
    required_cols = ['RecipeName', 'Ingredients', 'TotalTimeInMins', 'Course', 'Diet']
    
    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"The following required columns are missing from the dataset: {missing_cols}")

    # Drop rows where key columns are missing
    df.dropna(subset=required_cols, inplace=True)
    print("Dataset loaded and validated successfully.")
    print("Columns:", df.columns.to_list())

except Exception as e:
    print(f"\nFATAL ERROR loading or processing the dataset: {e}")
    print("The script will now exit as it cannot proceed without valid data.")
    sys.exit() # Exit the script if data loading fails

# --- 2. Feature Engineering & Preprocessing ---

# Function to clean the new, more complex 'Ingredients' column
def clean_ingredients(text):
    if not isinstance(text, str):
        return ""
    # Remove text in parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove quantities and units (e.g., "6", "to taste", "1 tablespoon")
    text = re.sub(r'[\d\./]+(\s*(tablespoon|teaspoon|cup|grams|kg|ml)\w*)?', '', text)
    # Remove common instruction words
    text = text.replace('- deseeded', '').replace('- thinly sliced', '').replace('- to taste', '')
    # Split by comma, strip whitespace, and join back
    ingredients = [item.strip() for item in text.split(',') if item.strip()]
    return ', '.join(ingredients)

# ENHANCED: Function to map 'Course' to a simpler 'Meal_Type'
def get_meal_type(course):
    if not isinstance(course, str):
        return 'Unknown'
    course = course.lower()
    if any(keyword in course for keyword in ['breakfast', 'pancakes', 'smoothie']):
        return 'Breakfast'
    if any(keyword in course for keyword in ['snack', 'appetizer', 'starter']):
        return 'Snack'
    if any(keyword in course for keyword in ['dessert', 'cakes', 'sweets']):
        return 'Dessert'
    # Default most main/side dishes to Lunch/Dinner
    if any(keyword in course for keyword in ['main course', 'side dish', 'lunch', 'dinner', 'biryani', 'pulao', 'dal', 'sabzi', 'curries', 'one pot dish', 'salad', 'soup']):
        return 'Dinner' # Treat Lunch and Dinner as the same category for broader results
    return 'Dinner' # Default

df['Cleaned_Ingredients'] = df['Ingredients'].apply(clean_ingredients)
df['Meal_Type'] = df['Course'].apply(get_meal_type)

# Vectorizer for finding ingredient similarity
vectorizer = TfidfVectorizer(stop_words='english')
# We create a combined feature for TF-IDF to learn from, including diet and course for context
df['Combined_Features'] = df['Cleaned_Ingredients'] + ' ' + df['Diet'] + ' ' + df['Course']
feature_matrix = vectorizer.fit_transform(df['Combined_Features'])

print("\nPreprocessing complete. Enhanced 'Meal_Type' mapping applied.")


# --- 3. The Recommendation Function ---

def get_meal_recommendations(height_cm, weight_kg, meal_type, weight_goal, desired_ingredients, user_allergies, max_cooking_time=None, diet_preference=None, num_results=10):
    """
    Generates meal recommendations using the new dataset structure and enhanced logic.
    """
    print("\n--- Starting Recommendation Process ---")
    
    # --- Stage 1: Initial Candidate Filtering ---
    if meal_type.lower() in ['lunch', 'dinner']:
        meal_mask = df['Meal_Type'].isin(['Dinner'])
    else:
        meal_mask = (df['Meal_Type'].str.lower() == meal_type.lower())

    # Start with a base set of candidates after meal type filtering
    base_candidate_df = df[meal_mask].copy()

    # Filter by allergies
    if user_allergies and user_allergies.lower() != 'none':
        allergens = [a.strip() for a in user_allergies.lower().split(',')]
        allergy_mask = base_candidate_df['Cleaned_Ingredients'].apply(
            lambda x: not any(allergen in x.lower() for allergen in allergens)
        )
        base_candidate_df = base_candidate_df[allergy_mask]

    # Filter by cooking time
    if max_cooking_time is not None:
        base_candidate_df = base_candidate_df[base_candidate_df['TotalTimeInMins'] <= max_cooking_time]

    # --- UPDATED: Strict Ingredient Filtering with Fallback ---
    candidate_df = base_candidate_df 

    if desired_ingredients and desired_ingredients.lower().strip() != 'none':
        must_have_ingredients = [ing.strip() for ing in desired_ingredients.lower().split(',') if ing.strip()]
        
        # Create a mask that checks if all must-have ingredients are in the recipe's ingredient list
        strict_mask = base_candidate_df['Cleaned_Ingredients'].apply(
            lambda x: all(ing in x.lower() for ing in must_have_ingredients)
        )
        strict_filtered_df = base_candidate_df[strict_mask]

        # If strict filtering yields results, use them. Otherwise, fall back to relaxed search.
        if not strict_filtered_df.empty:
            candidate_df = strict_filtered_df.copy() # Use a copy to avoid warnings
        else:
            print("[!] No recipes found with ALL desired ingredients. Relaxing search to find recipes with ANY of the ingredients.")
            relaxed_mask = base_candidate_df['Cleaned_Ingredients'].apply(
                lambda x: any(ing in x.lower() for ing in must_have_ingredients)
            )
            candidate_df = base_candidate_df[relaxed_mask].copy() # Use a copy to avoid warnings


    if candidate_df.empty:
        print("No recipes found after initial filtering. Consider relaxing constraints or desired ingredients.")
        return pd.DataFrame()
    print(f"Found {len(candidate_df)} candidates after all filtering.")

    # --- Stage 2: Ranking Candidates ---
    user_query = f"{desired_ingredients} {diet_preference or ''} {weight_goal}"
    user_vector = vectorizer.transform([user_query])
    candidate_matrix = vectorizer.transform(candidate_df['Combined_Features'])
    
    # FIX: Assign new columns using .loc to prevent SettingWithCopyWarning
    candidate_df.loc[:, 'similarity_score'] = cosine_similarity(user_vector, candidate_matrix).flatten()

    # ENHANCED: More advanced ranking score calculation
    def calculate_ranking_score(row):
        score = row['similarity_score'] * 1.0
        diet = row['Diet'].lower()

        # Big bonus for matching a specific diet preference
        if diet_preference and diet_preference.lower() in diet:
            score += 0.5
        
        # Bonus based on weight goal
        if weight_goal == 'Lose' and any(d in diet for d in ['diabetic friendly', 'sugar free diet', 'no onion no garlic']):
            score += 0.2
        if weight_goal == 'Gain' and any(d in diet for d in ['high protein']):
            score += 0.2
        
        return score

    candidate_df.loc[:, 'ranking_score'] = candidate_df.apply(calculate_ranking_score, axis=1)
    recommendations = candidate_df.sort_values(by='ranking_score', ascending=False).head(num_results)
    
    if recommendations.empty:
        print("Could not find any suitable recommendations after ranking.")
        return pd.DataFrame()

    print("--- Recommendations Generated ---")
    return recommendations

# --- 4. Example Usage ---
# This part is for testing and won't be called by the FastAPI backend
if __name__ == '__main__':
    print("\n" + "="*50)
    print("Example: User wants to Lose weight, looking for a vegetarian Dinner.")
    print("="*50)

    recommendations = get_meal_recommendations(
        height_cm=160,
        weight_kg=70,
        meal_type='Dinner',
        weight_goal='Lose',
        desired_ingredients='paneer',
        user_allergies='nuts',
        diet_preference='Vegetarian'
    )
    
    # FIX: Check if the recommendations DataFrame is empty before printing to prevent crash
    if not recommendations.empty:
        print(recommendations[['RecipeName', 'ranking_score', 'Diet', 'Course', 'Cleaned_Ingredients']])
    else:
        print("\n--> Final Result: No recommendations found matching the criteria.")
