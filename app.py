import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2 # OpenCV for Face Detection
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------
# We stick with the CORRECT spelling 'wrinkles' here.
# This ensures it matches the column names in your CSV database.
CNN_LABEL_MAP = {
    0: 'acne',
    1: 'dry',
    2: 'normal',
    3: 'oily',
    4: 'wrinkles' 
}

# Thresholds
MIN_FACE_RATIO = 0.15  # Face must be at least 15% of the screen
CONFIDENCE_THRESHOLD = 40.0 # If top prediction is lower than this, return Unknown

# -----------------------------------------------------------------
# LOADING ASSETS
# -----------------------------------------------------------------
@st.cache_resource
def load_models_and_assets():
    st.write("Loading AI models... This may take a moment.")
    
    # 1. Load CNN Model
    try:
        cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
    except Exception as e:
        st.error(f"Error loading cnn_model.h5: {e}")
        return None, None, None

    # 2. Load Product Database
    try:
        df = pd.read_csv("products_with_scores.csv")
    except FileNotFoundError:
        st.error("Error: 'products_with_scores.csv' not found. Run precompute_scores.py first.")
        return None, None, None

    # 3. Load Face Detector
    # Uses the default OpenCV XML for frontal faces
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    st.success("System Ready!")
    return cnn_model, df, face_cascade

# -----------------------------------------------------------------
# IMAGE PROCESSING LOGIC
# -----------------------------------------------------------------
def process_and_validate_image(image, face_cascade):
    """
    Detects face, draws box, and crops the face for the AI.
    """
    img_array = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    
    # CHECK: No face found
    if len(faces) == 0:
        return False, "No face detected. Ensure good lighting.", img_array, None

    # CHECK: Multiple faces (Take the largest)
    if len(faces) > 1:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[:1]

    (x, y, w, h) = faces[0]
    
    # CHECK: Face too small/far away
    img_h, img_w, _ = img_array.shape
    face_ratio = h / img_h
    if face_ratio < MIN_FACE_RATIO:
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 3) # Red Box
        return False, "Face too far. Move closer.", img_array, None

    # --- SUCCESS ---
    cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 3) # Green Box
    
    # IMPORTANT: Increase Padding to include forehead and cheeks
    pad_x = int(w * 0.25) 
    pad_y = int(h * 0.25)
    
    y1 = max(0, y - pad_y)
    y2 = min(img_h, y + h + pad_y)
    x1 = max(0, x - pad_x)
    x2 = min(img_w, x + w + pad_x)
    
    cropped_face = np.array(image.convert('RGB'))[y1:y2, x1:x2]
    
    return True, "Face detected.", img_array, cropped_face

def predict_skin_type(cnn_model, face_image, label_map):
    # Resize and Preprocess
    img_tensor = tf.convert_to_tensor(face_image, dtype=tf.float32)
    img_resized = tf.image.resize(img_tensor, (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    # Predict
    prediction = cnn_model.predict(img_batch)
    probabilities = prediction[0] # Get first item in batch
    
    # Get Top Prediction
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index] * 100
    
    # CHECK: Confidence Threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return "uncertain", confidence, probabilities
        
    skin_type_label = label_map.get(predicted_index, "Unknown")
    return skin_type_label, confidence, probabilities

def get_recommendations(product_df, predicted_skin_type):
    if predicted_skin_type == "uncertain":
        return pd.DataFrame()
        
    score_column = f"score_{predicted_skin_type}"
    if score_column not in product_df.columns:
        return pd.DataFrame()

    recommended_df = product_df[product_df[score_column] > 0.5]
    recommended_df = recommended_df.sort_values(by=score_column, ascending=False)
    return recommended_df.head(5)

# -----------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------
def main():
    st.title("✨ DermaClust: AI Skincare Analysis")
    st.markdown("Ensure good lighting and remove glasses for best results.")

    cnn_model, df, face_cascade = load_models_and_assets()
    if not all([cnn_model, df is not None, face_cascade]): st.stop()

    # Tabs
    tab1, tab2 = st.tabs(["📷 Upload", "📸 Camera"])
    image_to_process = None
    
    with tab1:
        uploaded = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded: image_to_process = Image.open(uploaded)
            
    with tab2:
        camera_photo = st.camera_input("Take a selfie")
        if camera_photo: image_to_process = Image.open(camera_photo)

    if image_to_process:
        st.divider()
        
        # 1. Validate Face Geometry
        with st.spinner("Scanning face geometry..."):
            is_valid, msg, boxed_img, cropped_face = process_and_validate_image(image_to_process, face_cascade)
        
        # Show the user the "Computer Vision" view
        st.image(boxed_img, caption="Analysis View", width=400)
        
        if not is_valid:
            st.error(f"⚠️ {msg}")
        else:
            # 2. Predict Skin Type
            with st.spinner("Analyzing skin texture..."):
                skin_type, confidence, all_probs = predict_skin_type(cnn_model, cropped_face, CNN_LABEL_MAP)
            
            # 3. Display Detailed Stats
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if skin_type == "uncertain":
                    st.warning("🤔 Uncertain")
                    st.caption(f"Top guess was {CNN_LABEL_MAP[np.argmax(all_probs)].capitalize()}, but confidence was low.")
                else:
                    st.metric("Detected", skin_type.capitalize())
                    st.metric("Confidence", f"{confidence:.1f}%")
            
            with col2:
                st.caption("AI Confidence Distribution:")
                # Create a bar chart of probabilities
                probs_df = pd.DataFrame({
                    'Skin Type': list(CNN_LABEL_MAP.values()),
                    'Probability': all_probs * 100
                })
                st.bar_chart(probs_df.set_index('Skin Type'))

            # 4. Recommendations
            st.divider()
            if skin_type != "uncertain":
                st.subheader(f"🧴 Recommendations for {skin_type.capitalize()}")
                results = get_recommendations(df, skin_type)
                
                if not results.empty:
                    for _, row in results.iterrows():
                        with st.container():
                            st.markdown(f"**{row['product_name']}**")
                            st.markdown(f"Type: {row['product_type']} | Price: ${row['product_price']}")
                            with st.expander("Ingredients"):
                                st.text(row['clean_ingreds'])
                            st.divider()
                else:
                    st.info("No specific products found in database.")
            else:
                st.info("Please try another photo. Ensure your face is evenly lit.")

if __name__ == "__main__":
    main()