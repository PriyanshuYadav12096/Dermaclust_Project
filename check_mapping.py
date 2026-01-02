import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset
# MAKE SURE THIS MATCHES YOUR ACTUAL PATH
dataset_path = os.path.join("dataset", "faceskin")

print(f"Checking folders in: {dataset_path}")

if not os.path.exists(dataset_path):
    print("❌ Error: Dataset folder not found.")
    print("Please make sure your folder is named 'dataset' and has 'faceskin' inside it.")
    exit()

# Create a data generator (just like training)
datagen = ImageDataGenerator()

try:
    # This looks at the folders and assigns the numbers
    generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    print("\n--- ✅ CORRECT MAPPING FOUND ---")
    print("Copy this dictionary EXACTLY into your app.py file:")
    print("---------------------------------------------------")
    
    # Get the class indices (e.g., {'acne': 0, 'dry': 1})
    class_indices = generator.class_indices
    
    # Swap keys and values to make it ready for app.py (e.g., {0: 'acne'})
    id_to_label = {v: k for k, v in class_indices.items()}
    
    print("CNN_LABEL_MAP = {")
    for index, label in id_to_label.items():
        print(f"    {index}: '{label}',")
    print("}")
    print("---------------------------------------------------")

except Exception as e:
    print(f"❌ Error reading directory: {e}")