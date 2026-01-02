import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
import os
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

print("--- Starting Pre-computation Process ---")

# --- 1. Load All Assets ---
print("Step 1/4: Loading models, tokenizer, and binarizer...")
try:
    bert_model = TFDistilBertForSequenceClassification.from_pretrained('models/bert_model_hf')
    tokenizer = DistilBertTokenizer.from_pretrained('assets/tokenizer_hf')
    with open('assets/mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    df = pd.read_csv("final_product_database.csv")
    print("Assets loaded.")
except Exception as e:
    print(f"Error loading assets: {e}")
    print("Please make sure you have run train_models.py successfully.")
    exit()

# --- 2. Get Ingredients ---
print("Step 2/4: Preparing ingredient list...")
all_ingredients = df['clean_ingreds'].astype(str).tolist()
if not all_ingredients:
    print("Error: No ingredients found in database.")
    exit()

# --- 3. Run Batch Prediction (Just like in app.py) ---
print("Step 3/4: Running batch predictions (this will take a while)...")
batch_size = 32  # Process 32 products at a time
all_logits = []
total_batches = (len(all_ingredients) // batch_size) + 1

for i in range(0, len(all_ingredients), batch_size):
    print(f"  ...processing batch {i//batch_size + 1} of {total_batches}")
    batch_ingredients = all_ingredients[i : i + batch_size]

    tokenized_inputs = tokenizer(
        batch_ingredients,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    try:
        predictions = bert_model(tokenized_inputs)
        batch_logits = predictions.logits
        all_logits.append(batch_logits)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        exit()

# Combine all batch results
print("  ...combining batch results.")
logits = tf.concat(all_logits, axis=0)
probabilities = tf.nn.sigmoid(logits).numpy()

# --- 4. Save Scores to New CSV ---
print("Step 4/4: Saving scores to new CSV...")
label_names = mlb.classes_ # e.g., ['acne', 'dry', 'normal', 'oily', 'wrinkles']

# Create a new dataframe with just the scores
score_df = pd.DataFrame(probabilities, columns=[f"score_{label}" for label in label_names])

# Combine the original dataframe with the new scores dataframe
df_with_scores = pd.concat([df, score_df], axis=1)

# Save the final file
output_filename = "products_with_scores.csv"
df_with_scores.to_csv(output_filename, index=False)

print("\n--- Pre-computation Complete! ---")
print(f"New file created: {output_filename}")
print("You can now run the Streamlit app.")