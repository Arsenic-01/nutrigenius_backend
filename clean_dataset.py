import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return text

    # Fix common encoding issues
    text = text.replace('Â', '')
    text = text.replace('\xa0', ' ')  # non-breaking space
    text = text.replace('â€™', "'")   # smart quotes
    text = text.replace('â€“', '-')   # en-dash
    text = text.replace('â€œ', '"').replace('â€�', '"')  # quotes
    text = text.replace('Ã', 'a')  # malformed accented characters

    # Fix spacing and slashes
    text = re.sub(r'(\d+)\s*/\s*(\d+)', lambda m: str(round(int(m.group(1)) / int(m.group(2)), 2)), text)  # e.g., 1 / 2 → 0.5
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text.strip()

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)

    # Apply cleaning to all string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(clean_text)

    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned dataset saved to: {output_file}")

# Usage
if __name__ == "__main__":
    input_csv = "NutriGeniusDataset.csv"         # Change to your file name
    output_csv = "NutriGeniusDataset.csv"
    clean_dataset(input_csv, output_csv)
