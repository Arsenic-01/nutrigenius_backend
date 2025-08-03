import pandas as pd
from rapidfuzz import process, fuzz
import re
import time
import sys

def create_master_dataset():
    """
    Loads the core Indian recipe and nutritional datasets, performs cleaning,
    merging, and feature engineering, and saves a single, optimized master
    dataset for the recommendation model.
    """
    start_time = time.time()
    print("Starting creation of the master dataset. This may take a few minutes...")

    # --- 1. Load Focused Datasets ---
    try:
        # Load your primary recipe dataset
        df = pd.read_csv('./datasets/NutriGeniusDataset.csv')
        df.dropna(subset=['RecipeName', 'Ingredients', 'TotalTimeInMins', 'Course', 'Diet', 'Cuisine'], inplace=True)
        print("✅ Recipe dataset loaded.")

        # Load the Indian nutritional dataset
        nutrition_df = pd.read_csv('./datasets/IndianFoodNutrition.csv')
        nutrition_df.rename(columns={'Dish Name': 'RecipeName'}, inplace=True)
        print("✅ Indian nutritional dataset loaded.")

    except FileNotFoundError as e:
        print(f"❌ ERROR: A required raw data file was not found: {e.filename}.")
        sys.exit()

    # --- 2. Perform Cleaning and Feature Engineering ---
    print("\nCleaning ingredients and creating new features...")
    def clean_ingredients(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'[\d\./]+(\s*(tablespoon|teaspoon|cup|grams|kg|ml)\w*)?', '', text)
        ingredients = [item.strip() for item in text.split(',') if item.strip()]
        return ', '.join(ingredients)

    df['Cleaned_Ingredients'] = df['Ingredients'].apply(clean_ingredients)
    df['Meal_Type'] = df['Course'].apply(lambda c: 'Breakfast' if 'breakfast' in str(c).lower() else 'Dinner')
    df['Ingredient_Count'] = df['Cleaned_Ingredients'].apply(lambda x: len(x.split(',')))

    # --- 3. Fuzzy Merge Datasets ---
    print("Merging nutritional data using fuzzy matching...")
    
    recipe_names = df['RecipeName'].tolist()
    nutrition_names = nutrition_df['RecipeName'].tolist()
    
    name_mapping = {}
    for name in recipe_names:
        # Find the best match with a reasonable confidence score
        match = process.extractOne(name, nutrition_names, scorer=fuzz.WRatio, score_cutoff=80)
        if match:
            # match is a tuple: (matched_string, score, index)
            name_mapping[name] = match[0]

    df['match_name'] = df['RecipeName'].map(name_mapping)
    
    # Merge the dataframes based on this fuzzy mapping
    df = pd.merge(df, nutrition_df, left_on='match_name', right_on='RecipeName', how='left', suffixes=('', '_nutr'))
    df.drop(columns=['match_name', 'RecipeName_nutr'], inplace=True)

    # --- 4. Save the Final Master Dataset ---
    df.to_csv('master_recipe_dataset.csv', index=False)
    end_time = time.time()
    
    print(f"\n🎉 Master dataset created successfully in {end_time - start_time:.2f} seconds!")
    print("Saved as 'master_recipe_dataset.csv'.")

if __name__ == '__main__':
    create_master_dataset()
