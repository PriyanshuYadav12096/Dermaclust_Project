# --- Make sure these imports are at the TOP of your file ---
import pandas as pd
import ast
import numpy as np
import pickle
import os
import tensorflow as tf # <-- Make sure TF is imported
from ingredients import KEY_INGREDIENTS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
# --- REMOVE OLD KERAS TEXT IMPORTS ---
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import Input, Embedding, LayerNormalization, MultiHeadAttention
# from tensorflow.keras.models import Model (keep if you need, but Transformer model is different)

# --- ADD NEW TRANSFORMER IMPORTS ---
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from tensorflow.keras.optimizers import Adam

# --- Keep your CNN imports ---
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model # <-- Need this for the CNN
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("--- Starting Model Training Process (with Transformer) ---")
# --- STEP 1: LOAD AND PREPARE TEXT DATA ---
print("Step 1/5: Loading and preparing text data...")
try:
    df = pd.read_csv("final_product_database.csv")
    df['ingredients'] = df['clean_ingreds'].apply(lambda x: ast.literal_eval(x.lower()))
    df['ingredients_text'] = df['ingredients'].apply(lambda x: ' '.join(x))

    # --- INGREDIENT-BASED LABELING LOGIC ---
    def assign_labels_by_ingredient(ingredient_list):
        assigned = set()
        for label, keywords in KEY_INGREDIENTS.items():
            for keyword in keywords:
                if keyword in ingredient_list:
                    assigned.add(label)
        if not assigned:
            return ['normal']
        return list(assigned)

    df['label_tags'] = df['ingredients'].apply(assign_labels_by_ingredient)
    # Define the master list of all possible labels
    labels = ['normal', 'oily', 'dry', 'acne', 'wrinkles']
    
    print("Text data loaded and labeled using the complete ingredient list.")
except FileNotFoundError:
    print("\nERROR: 'final_product_database.csv' not found.")
    print("Please make sure the Excel file is in the same folder as this script.")
    exit()

# --- STEP 2: TOKENIZE TEXT AND TRAIN TRANSFORMER MODEL ---
print("\nStep 2/5: Building and fine-tuning the ingredients (DistilBERT) model...")

# 1. Load Pre-trained Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 2. Tokenize Your Data
# This is different from Keras. It creates 'input_ids' and 'attention_mask'
X = tokenizer(
    df['ingredients_text'].tolist(),
    add_special_tokens=True,
    max_length=128,  # You can tune this
    padding='max_length',
    truncation=True,
    return_tensors='tf'
)

# 3. Binarize Labels (Same as before)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['label_tags'])
# We need to save the mlb classes, not the 'labels' list
# The order might be different!
labels_list_from_mlb = mlb.classes_

# 4. Create Train/Validation Split
# 4. Create Train/Validation Split
# Convert TF Tensors to NumPy arrays for scikit-learn
input_ids_np = X['input_ids'].numpy()
attention_mask_np = X['attention_mask'].numpy()

# Now, split the NumPy arrays
X_train_ids, X_val_ids, y_train, y_val = train_test_split(input_ids_np, y, test_size=0.2, random_state=42)
X_train_mask, X_val_mask, _, _ = train_test_split(attention_mask_np, y, test_size=0.2, random_state=42)
# We need to split both input_ids and attention_mask
# X_train_ids, X_val_ids, y_train, y_val = train_test_split(X['input_ids'], y, test_size=0.2, random_state=42)
# X_train_mask, X_val_mask, _, _ = train_test_split(X['attention_mask'], y, test_size=0.2, random_state=42)

# Create TF-friendly dictionaries for training and validation data
X_train = {'input_ids': X_train_ids, 'attention_mask': X_train_mask}
X_val = {'input_ids': X_val_ids, 'attention_mask': X_val_mask}

# 5. Load Pre-trained Model
bert_model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(labels_list_from_mlb),
    problem_type="multi_label_classification" # <-- Crucial for multi-label
)

# 6. Compile the Model for Fine-Tuning
# We must use a lower learning rate for fine-tuning
bert_model.compile(
    optimizer=Adam(learning_rate=3e-5),
    # Use BCEWithLogitsLoss because the model outputs raw logits
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 7. Fine-Tune the Model
print("--- Fine-tuning DistilBERT model... ---")
bert_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=4, # Transformers fine-tune very quickly
    batch_size=16
)
print("Ingredients (DistilBERT) model fine-tuned successfully.")

# --- STEP 3: TRAIN SKIN ANALYZER (TRANSFER LEARNING) ---
print("\nStep 3/5: Building and training skin image model with Transfer Learning...")

# Use 224x224, the standard for MobileNetV2
img_size = (224, 224)
image_path = os.path.join("dataset", "faceskin")

if not os.path.exists(image_path):
    print(f"\nERROR: Image directory not found at '{image_path}'")
    print("Please ensure your 'faceskin' folder is inside a 'dataset' folder.")
    exit()

# Use MobileNetV2's specific preprocessing function
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # <--- CRITICAL CHANGE
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    image_path,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
   
)
print("--- IMPORTANT: CNN CLASS MAP ---")
print(train_gen.class_indices)
print("----------------------------------")

val_gen = datagen.flow_from_directory(
    image_path,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_gen.num_classes

# 1. Load the Base Model (MobileNetV2)
# We load the model pre-trained on 'imagenet'
# 'include_top=False' removes the final classification layer
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# 2. Freeze the Base Model
# We "freeze" the layers of the base model so they don't get changed
# during the first phase of training.
base_model.trainable = False

# 3. Add Your Custom "Head"
# We build a new model, adding our own layers on top of the base model
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False) # 'training=False' is important here
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

image_model = Model(inputs, outputs)

# 4. Compile the Model (Phase 1)
image_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Define Callbacks
# This will stop training when 'val_accuracy' stops improving
early_stopper = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# 6. Train the Head
print("--- Training the model head (Phase 1) ---")
history = image_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25, # Set high, EarlyStopping will find the best one
    callbacks=[early_stopper]
)

print("--- Phase 1 complete. Now fine-tuning... ---")

# 7. Fine-Tuning (Phase 2)
# Now, we "unfreeze" the top layers of the base model
base_model.trainable = True

# Let's freeze the first 100 layers and only fine-tune the rest
for layer in base_model.layers[:100]:
    layer.trainable = False

# 8. Re-compile the Model for Fine-Tuning
# We MUST use a very low learning rate for fine-tuning
image_model.compile(
    optimizer=Adam(learning_rate=1e-5), # <--- VERY LOW learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 9. Train Again (Fine-Tuning)
# We continue training from where we left off
fine_tune_epochs = 20
total_epochs = 25 + fine_tune_epochs

history_fine_tune = image_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], # Start from last epoch
    callbacks=[early_stopper] # Use the same early stopper
)

print("Skin image model (MobileNetV2) trained successfully.")


# --- STEP 4: SAVE EVERYTHING ---
print("\nStep 4/5: Saving all models and assets...")

os.makedirs('models', exist_ok=True)
os.makedirs('assets', exist_ok=True)

# 1. Save the Transformer Model (New Way)
bert_model.save_pretrained('models/bert_model_hf')
print("Hugging Face Transformer model saved to 'models/bert_model_hf'.")

# 2. Save the Transformer Tokenizer (New Way)
tokenizer.save_pretrained('assets/tokenizer_hf')
print("Hugging Face Tokenizer saved to 'assets/tokenizer_hf'.")

# 3. Save the CNN Model (Same as before)
image_model.save('models/cnn_model.h5')
print("CNN model saved to 'models/cnn_model.h5'.")

# 4. Save the Label Binarizer (CRITICAL CHANGE)
# We must save the MultiLabelBinarizer (mlb) because it knows the
# exact order of the output labels.
with open('assets/mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)
print("Label Binarizer (mlb) saved to 'assets/mlb.pkl'.")
# (We don't need 'labels.pkl' anymore)


# --- STEP 5: FINISH ---
print("\n--- Model Training Process Finished ---")
print("You can now run your Streamlit app with the command: streamlit run app.py")