import pandas as pd
import time
from ddgs import DDGS
from tqdm import tqdm

# Path to your dataset CSV
CSV_FILE = "NutriGeniusDataset.csv"

# Load your existing CSV
df = pd.read_csv(CSV_FILE)

# Ensure 'image-url' column exists
if 'image-url' not in df.columns:
    df['image-url'] = ""

# DuckDuckGo image search function
def fetch_image_url(query):
    with DDGS() as ddgs:
        results = ddgs.images(query + " Indian food", max_results=1)
        for r in results:
            return r["image"]
    return None

print(f"🔎 Fetching images for {len(df)} recipes...\n")
failed_rows = []

# Iterate with progress bar
for i, row in tqdm(df.iterrows(), total=len(df), desc="🔄 Processing"):
    image_url = str(row['image-url']) if not pd.isna(row['image-url']) else ""

    # Only fetch if image URL is missing or from archanaskitchen.com
    if not image_url or "archanaskitchen.com" in image_url:
        try:
            recipe_name = str(row['RecipeName'])
            print(f"[{i+1}/{len(df)}] Searching image for: {recipe_name}")

            fetched_url = fetch_image_url(recipe_name)

            if fetched_url:
                df.at[i, 'image-url'] = fetched_url
                print(f"✅ Fetched: {fetched_url}")
            else:
                print("⚠️ No image found.")
                failed_rows.append((i, recipe_name))

            # Save after every row
            df.to_csv(CSV_FILE, index=False)

            # Sleep to avoid rate limits
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error at row {i+1}: {e}")
            failed_rows.append((i, recipe_name))
            continue

# Final save just in case
df.to_csv(CSV_FILE, index=False)

# Log failures
if failed_rows:
    print("\n⚠️ Failed to fetch image URLs for:")
    for idx, name in failed_rows:
        print(f"  Row {idx+1}: {name}")

print("\n🎉 Image fetching completed and saved to:", CSV_FILE)
