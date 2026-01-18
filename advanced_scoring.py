import pandas as pd
import ast
from ingredients import KEY_INGREDIENTS, COMEDOGENIC_REAGENTS, POTENTIAL_IRRITANTS

def calculate_scientific_score(ingredient_list, skin_type):
    score = 0.0
    ing_set = set([i.strip().lower() for i in ingredient_list])
    
    # 1. POSITIVE MATCHES (+2 for 'best', +1 for 'good')
    # (Note: You can refine your ingredients.py to have 'best' vs 'good' categories)
    for match in ing_set:
        if match in KEY_INGREDIENTS.get(skin_type, []):
            score += 1.5  # Heavy weight for known beneficial ingredients
            
    # 2. NEGATIVE FILTERS (The "Winner" Logic)
    if skin_type in ['acne', 'oily']:
        for bad in COMEDOGENIC_REAGENTS:
            if bad in ing_set:
                score -= 2.0  # Penalize heavily for clogging pores
                
    if skin_type == 'dry':
        for irritant in POTENTIAL_IRRITANTS:
            if irritant in ing_set:
                score -= 1.5  # Penalize for drying alcohols/fragrance
                
    return max(0, score) # Ensure score doesn't go below 0

# Load your database
df = pd.read_csv("final_product_database.csv")
df['clean_ingreds'] = df['clean_ingreds'].apply(ast.literal_eval)

# Apply scoring for all types
for s_type in ['acne', 'oily', 'dry', 'normal', 'wrinkles']:
    df[f'score_{s_type}'] = df['clean_ingreds'].apply(lambda x: calculate_scientific_score(x, s_type))
    # Normalize score between 0 and 1
    max_val = df[f'score_{s_type}'].max()
    if max_val > 0:
        df[f'score_{s_type}'] = df[f'score_{s_type}'] / max_val

df.to_csv("products_with_scores.csv", index=False)
print("✅ Advanced Database Created with Scientific Weighting!")