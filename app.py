import streamlit as st
import pandas as pd
import mediapipe as mp
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2 
from assessment_logic import calculate_hybrid_profile
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ingredients import KEY_INGREDIENTS
# -----------------------------------------------------------------
# INITIALIZE MEDIAPIPE
# -----------------------------------------------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# -----------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------
# NOTE: Verify these match your 'dataset/faceskin' folder order (alphabetical)
CNN_LABEL_MAP = {
    0: 'acne',
    1: 'dry',
    2: 'normal',
    3: 'oily',
    4: 'wrinkles' 
}

# -----------------------------------------------------------------
# LOADING ASSETS
# -----------------------------------------------------------------
# @st.cache_resource
# def load_models_and_assets():
#     try:
#         cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
#         df = pd.read_csv("products_with_scores.csv")
#         return cnn_model, df
#     except Exception as e:
#         st.error(f"Error loading assets: {e}")
#         return None, None
@st.cache_resource
def load_models_and_assets():
    try:
        cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
        df = pd.read_csv("products_with_scores.csv")
        
        # NEW: Load the dynamic label map from training
        import json
        with open('assets/label_map.json', 'r') as f:
            raw_map = json.load(f)
            # Convert keys back to integers
            label_map = {int(k): v for k, v in raw_map.items()}
            
        return cnn_model, df, label_map
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None
# -----------------------------------------------------------------
# IMAGE PROCESSING FUNCTIONS
# -----------------------------------------------------------------
def extract_skin_zones(img_array):
    results = face_mesh.process(img_array)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = img_array.shape
    def get_ptr(idx): return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

    # Extract Forehead, Left Cheek, Right Cheek
    fh_x, fh_y = get_ptr(10)
    lc_x, lc_y = get_ptr(234)
    rc_x, rc_y = get_ptr(454)

    return {
        "forehead": img_array[max(0, fh_y-50):fh_y+50, max(0, fh_x-50):fh_x+50],
        "left_cheek": img_array[max(0, lc_y-50):lc_y+50, max(0, lc_x-50):lc_x+50],
        "right_cheek": img_array[max(0, rc_y-50):rc_y+50, max(0, rc_x-50):rc_x+50]
    }

def process_and_validate_image(image):
    img_array = np.array(image.convert('RGB'))
    img_h, img_w, _ = img_array.shape
    results = face_detection.process(img_array)
    
    if not results.detections:
        return False, "No face detected.", img_array, None

    bbox = results.detections[0].location_data.relative_bounding_box
    x, y, w, h = int(bbox.xmin * img_w), int(bbox.ymin * img_h), int(bbox.width * img_w), int(bbox.height * img_h)

    # Draw Green Box for UI
    cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    pad_w, pad_h = int(w * 0.2), int(h * 0.2)
    cropped_face = np.array(image.convert('RGB'))[max(0, y-pad_h):min(img_h, y+h+pad_h), max(0, x-pad_w):min(img_w, x+w+pad_w)]
    
    return True, "Face verified.", img_array, cropped_face

# -----------------------------------------------------------------
# AI PREDICTION LOGIC
# -----------------------------------------------------------------
def enhance_skin_texture(img_array):
    """
    Normalizes lighting and enhances skin texture using CLAHE.
    This helps the AI see 'oil' and 'acne' regardless of lighting conditions.
    """
    # Convert to YUV color space (better for lighting correction)
    img_yuv = cv2.cvtColor(np.array(img_array), cv2.COLOR_RGB2YUV)
    
    # Create a CLAHE object (arguments are for clip limit and grid size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Apply CLAHE to the Y channel (the brightness/luma channel)
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    
    # Convert back to RGB
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

def predict_skin_type(cnn_model, img, label_map):
    # --- NEW: Enhance the image first ---
    img_enhanced = enhance_skin_texture(img)
    
    img_resized = tf.image.resize(tf.convert_to_tensor(img_enhanced, dtype=tf.float32), (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    prediction = cnn_model.predict(np.expand_dims(img_preprocessed, axis=0))[0]
    
    idx = np.argmax(prediction)
    return label_map.get(idx, "Unknown"), prediction[idx] * 100, prediction

def predict_zone_type(cnn_model, zone_image, label_map):
    """
    Analyzes a specific patch (forehead or cheek).
    Uses a higher confidence threshold (65%) to ensure accuracy.
    """
    # Apply texture enhancement first
    zone_enhanced = enhance_skin_texture(zone_image)
    
    img_tensor = tf.convert_to_tensor(zone_enhanced, dtype=tf.float32)
    img_resized = tf.image.resize(img_tensor, (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    
    prediction = cnn_model.predict(np.expand_dims(img_preprocessed, axis=0))[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index] * 100
    
    # Strict threshold for patches: If AI is less than 65% sure, it's 'uncertain'
    if confidence < 65.0:
        return "uncertain", confidence
        
    return label_map.get(predicted_index, "Unknown"), confidence
# scan the product list and highlight the hero ingedients.
def get_hero_ingredients(ingredient_list, skin_type):
    """
    Identifies which ingredients in a product actually match 
    the beneficial list for the detected skin type.
    """
    if skin_type not in KEY_INGREDIENTS:
        return []
        
    beneficial_set = set(KEY_INGREDIENTS[skin_type])
    # Find the intersection
    matches = [ing for ing in ingredient_list if ing.strip().lower() in beneficial_set]
    return matches[:3] # Return the top 3 matches for brevity

def get_recommendations(product_df, skin_type):
    target_col = f"score_{skin_type}"
    if target_col not in product_df.columns:
        target_col = "score_normal" 
        
    # Filter products that match the skin type (Score > 0.3)
    matches = product_df[product_df[target_col] > 0.3].copy()
    matches = matches.sort_values(by=target_col, ascending=False)
    
    # Helper to find specific product categories
    def find_best(category_keywords):
        pattern = '|'.join(category_keywords)
        return matches[matches['product_type'].str.contains(pattern, case=False, na=False)].head(1)

    # Build the routine slots
    routine = {
        "AM": {
            "Step 1: Cleanse": find_best(['Cleanser', 'Wash']),
            "Step 2: Treat": find_best(['Serum', 'Toner', 'Essence']),
            "Step 3: Protect": find_best(['SPF', 'Sunscreen', 'Day Cream', 'Moisturizer'])
        },
        "PM": {
            "Step 1: Cleanse": find_best(['Cleanser', 'Wash', 'Oil']),
            "Step 2: Treat": find_best(['Serum', 'Treatment', 'Night']),
            "Step 3: Hydrate": find_best(['Moisturizer', 'Cream', 'Night Cream'])
        }
    }
    return routine
# -----------------------------------------------------------------
# MAIN INTERFACE
# -----------------------------------------------------------------
def main():
    st.set_page_config(page_title="DermaClust AI", layout="centered")
    st.title("✨ DermaClust: Professional Skin Analysis")
    
    cnn_model, df,label_map = load_models_and_assets()
    if cnn_model is None or df is None: st.stop()

    tab1, tab2 = st.tabs(["📷 Upload Photo", "📸 Use Camera"])
    img_input = None
    
    with tab1:
        uploaded = st.file_uploader("Upload selfie", type=["jpg", "png", "jpeg"])
        if uploaded: img_input = Image.open(uploaded)
    with tab2:
        camera = st.camera_input("Take selfie")
        if camera: img_input = Image.open(camera)
    # --- NEW: Clinical Questionnaire Section ---
    st.divider()
    st.subheader("📋 Clinical Verification")
    st.info("The AI is analyzing your photo, but your input helps reach 100% accuracy.")
    
    col1, col2 = st.columns(2)
    with col1:
        q_feel = st.selectbox(
            "How does your skin feel 1 hour after washing?", 
            ["Normal", "Tight/Itchy", "Greasy/Shiny", "Oil only in T-Zone"]
        )
    with col2:
        q_breakouts = st.selectbox(
            "How often do you experience breakouts?", 
            ["Rarely", "Occasionally", "Frequent"]
        )
    
    user_quiz = {
        'feel_after_wash': q_feel, 
        'breakouts': q_breakouts
    }
    if img_input:
        is_valid, msg, boxed_img, face_crop = process_and_validate_image(img_input)
        st.image(boxed_img, caption="Face Detection Status", width=450)
        
        if not is_valid:
            st.error(msg)
        else:
            # 1. ZONE ANALYSIS
            zones = extract_skin_zones(np.array(img_input.convert('RGB')))
            final_type = "uncertain"
            
            if zones:
                st.subheader("🔍 Localized Analysis")
                cols = st.columns(3)
                z_preds = {}
                for i, (name, patch) in enumerate(zones.items()):
                    cols[i].image(patch, caption=name.capitalize())
                    label, conf= predict_zone_type(cnn_model, patch, CNN_LABEL_MAP)
                    z_preds[name] = label
                
                # Logic Synthesis
                fh, lc = z_preds.get("forehead"), z_preds.get("left_cheek")
                if fh == "oily" and lc == "dry":
                    final_type = "combination"
                else:
    # Get raw AI predictions
                   _, confidence, probs = predict_skin_type(cnn_model, face_crop, CNN_LABEL_MAP)
    
    # Convert AI array to a dictionary for the hybrid logic
                ai_results = dict(zip(CNN_LABEL_MAP.values(), probs))
    
    # NEW: Merge AI results with User Quiz
                final_type, combined_weights = calculate_hybrid_profile(ai_results, user_quiz)
    
                st.write("### 📊 Hybrid Confidence (AI + Clinical)")
                chart_data = pd.DataFrame(
                     combined_weights.values(), 
                     index=combined_weights.keys(), 
                     columns=["Weighted Score"]
                       )
                st.bar_chart(chart_data)
                    # DEBUG CHART: Shows raw AI probability for each type
                st.write("### 📊 AI Confidence Breakdown")
                chart_data = pd.DataFrame(probs, index=CNN_LABEL_MAP.values(), columns=["Confidence"])
                st.bar_chart(chart_data)

            # 2. RECOMMENDATIONS
            if final_type != "uncertain":
                st.metric("Detected Profile", final_type.capitalize())
                st.divider()
                
                # If "combination", we fetch products for "normal" skin as it's the safest balance
                results = get_recommendations(df, final_type)
                
                st.subheader(f"🧴 Top Recommendations for {final_type.capitalize()} Skin")
                # app.py (Replace the recommendation display block)

            if final_type != "uncertain":
                st.metric("Detected Profile", final_type.capitalize())
                st.divider()
                
                routine = get_recommendations(df, final_type)
                
                st.subheader(f"✨ Recommended {final_type.capitalize()} Regimen")
                
                # Create two columns for AM and PM
                col_am, col_pm = st.columns(2)
                
                # app.py (Update the loop inside the AM and PM columns)

# --- Updated AM Display ---
                with col_am:
                    st.markdown("#### ☀️ Morning (AM)")
                    for step, prod_df in routine["AM"].items():
                        if not prod_df.empty:
                            row = prod_df.iloc[0]
                            with st.expander(f"{step}: {row['product_name']}"):
                                st.caption(f"**Type:** {row['product_type']}")
                                
                                # NEW: Explainability Feature
                                # Assuming 'clean_ingreds' is stored as a list or string of a list
                                ing_list = row['clean_ingreds']
                                if isinstance(ing_list, str):
                                    import ast
                                    ing_list = ast.literal_eval(ing_list)
                                
                                heroes = get_hero_ingredients(ing_list, final_type)
                                
                                if heroes:
                                    st.success(f"**Why it works:** Contains **{', '.join(heroes)}**, which are excellent for {final_type} skin.")
                                else:
                                    st.write(f"Matched based on overall {final_type} compatibility score.")
                        else:
                            st.write(f"*{step}: No match found.*")

                # --- Updated PM Display (Do the same for PM) ---
                with col_pm:
                    st.markdown("#### 🌙 Evening (PM)")
                    for step, prod_df in routine["PM"].items():
                        if not prod_df.empty:
                            row = prod_df.iloc[0]
                            with st.expander(f"{step}: {row['product_name']}"):
                                st.caption(f"**Type:** {row['product_type']}")
                                
                                # Explainability Feature
                                ing_list = row['clean_ingreds']
                                if isinstance(ing_list, str):
                                    import ast
                                    ing_list = ast.literal_eval(ing_list)
                                
                                heroes = get_hero_ingredients(ing_list, final_type)
                                
                                if heroes:
                                    st.success(f"**Why it works:** Contains **{', '.join(heroes)}**.")
                                else:
                                    st.write("Matches your profile's safety and efficacy requirements.")
                        else:
                            st.write(f"*{step}: No match found.*")




                    

if __name__ == "__main__":
    main()