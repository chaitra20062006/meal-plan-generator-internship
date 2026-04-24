import json
import pandas as pd
import pathlib

# This script is designed to take the official IFCT 2017 Dataset (once downloaded locally as a CSV)
# and convert it into the required JSON schema for the NutriGenie RAG backend.
# The online npm repository recently blocked public API scraping, so you need to provide the CSV.

def process_ifct_csv_to_json(csv_path="index.csv", output_path="indian_nutrition_dataset.json"):
    try:
        print(f"Loading IFCT dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        output_data = []
        
        for index, row in df.iterrows():
            item = {
                "food_id": f"IFCT-{str(index).zfill(3)}",
                "category": row.get("Food_group", "General"),
                "name_en": str(row.get("name", row.get("Food_name", "Unknown"))),
                "name_hi": row.get("Scientific_name", ""),
                "name_regional": {},
                "per_100g": {
                    # Note: You may need to adjust these column headers based on the exact IFCT headers
                    "calories": float(row.get("Energy", row.get("Energy_kcal", 0))),
                    "protein_g": float(row.get("Protein", 0)),
                    "carbs_g": float(row.get("Carbohydrate", 0)),
                    "fat_g": float(row.get("Fat", 0)),
                    "fiber_g": float(row.get("Dietary_Fiber", 0))
                },
                "micronutrients": {
                    "iron_mg": float(row.get("Iron", 0)),
                    "calcium_mg": float(row.get("Calcium", 0)),
                    "zinc_mg": float(row.get("Zinc", 0)),
                },
                "common_dishes": [str(row.get("name", "Ingredient"))],
                "allergen_tags": [],
                "season": "year-round"
            }
            output_data.append(item)
            
        print(f"Successfully mapped {len(output_data)} ingredients.")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"Dataset successfully exported to {output_path}!")
        print(f"Please move this file to backend/data/indian_nutrition_dataset.json to hot-swap the backend math.")

    except Exception as e:
        print(f"Extraction failed: {e}")
        print("Please ensure you have placed the official IFCT index.csv in the same directory.")

if __name__ == "__main__":
    process_ifct_csv_to_json()
