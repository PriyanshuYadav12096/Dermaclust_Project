# prepare_database.py (Final Version)
import pandas as pd
import re

# We are ONLY using the good, working dataset.
input_file = 'detailed_cosmetics.csv'
output_file = 'final_product_database.csv'

# Helper function to clean price data
def clean_price(price):
    if isinstance(price, str):
        match = re.search(r'[\d\.]+', price)
        if match: return float(match.group(0))
    if isinstance(price, (int, float)): return price
    return None

# Helper function to standardize ingredient lists
def standardize_ingredients(ingred_string):
    if not isinstance(ingred_string, str): return "[]"
    if ingred_string.strip().startswith('['): return ingred_string.lower()
    ingredients = [item.strip() for item in ingred_string.split(',')]
    return str(ingredients).lower()

try:
    print(f"Loading the dataset from '{input_file}'...")
    df = pd.read_csv(input_file)

    # 1. Rename columns to the format your app expects
    print("Standardizing column names...")
    df_renamed = df.rename(columns={
        'name': 'product_name', 'ingredients': 'clean_ingreds',
        'category': 'product_type', 'Price': 'product_price'
    })

    # 2. Select only the four columns you need
    df_final = df_renamed[['product_name', 'clean_ingreds', 'product_type', 'product_price']]

    # 3. Clean and standardize the data
    print("Cleaning and standardizing data...")
    df_final['clean_ingreds'] = df_final['clean_ingreds'].apply(standardize_ingredients)
    df_final['product_price'] = df_final['product_price'].apply(clean_price)

    # 4. Remove any rows with missing essential information
    df_final.dropna(subset=['product_name', 'clean_ingreds', 'product_type', 'product_price'], inplace=True)
    df_final.drop_duplicates(subset=['product_name'], inplace=True, keep='first')

    # 5. Save the final, clean database
    df_final.to_csv(output_file, index=False)

    print(f"\n✅ Success! Your database is ready: '{output_file}'")
    print(f"Total products available: {len(df_final)}")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")