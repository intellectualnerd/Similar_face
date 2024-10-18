import os
import streamlit as st 
import cv2
from PIL import Image
from mtcnn import MTCNN
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Initialize MTCNN for face detection and ResNet50 for feature extraction
detector = MTCNN()
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the embedding and filenames pkl files
embedding_path = os.path.join(current_dir, 'embedding.pkl')
filenames_path = os.path.join(current_dir, 'filenames.pkl')

# Load the feature list and filenames from pickle files
feature_list = np.array(pickle.load(open(embedding_path, "rb")))
filenames = pickle.load(open(filenames_path, "rb"))

# Ensure the uploads directory exists
UPLOAD_FOLDER = os.path.join(current_dir, "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to save uploaded image
def save_upload_image(uploaded_image):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_image.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        return file_path  # Return the file path of the saved image
    except Exception as e:
        st.error(f"Error saving the image: {e}")
        return None

# Function to extract features from the uploaded image
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    
    if results:
        x, y, width, height = results[0]['box']
    
        # Extract the face from the image
        face = img[y:y + height, x:x + width]

        # Convert the face to RGB for processing
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Prepare the face image for feature extraction
        image = Image.fromarray(face_rgb)  # Convert to PIL Image
        image = image.resize((224, 224))  # Resize to (224, 224)

        face_array = np.asarray(image)  # Convert to NumPy array
        face_array = face_array.astype("float32")  # Convert to float32

        # Expand dimensions to create a batch
        extracted_img = np.expand_dims(face_array, axis=0)
        # Preprocess the image for the model
        preprocessed_img = preprocess_input(extracted_img)

        # Predict features using the model
        result = model.predict(preprocessed_img).flatten()
        return result
    else:
        st.text("No face detected in the image.")
        return None

def recommend(feature_list, features):
    # Calculate cosine similarity with all features in the feature list
    similarity = []
    for feature in feature_list:
        sim = cosine_similarity(features.reshape(1, -1), feature.reshape(1, -1))
        similarity.append(sim[0][0])  # Get the similarity score

    # Find the index of the most similar face
    index_pos = sorted(enumerate(similarity), reverse=True, key=lambda x: x[1])[0][0]

    return index_pos

# Streamlit app title
st.title("Which Bollywood Celebrity Are You?")

# Upload image using Streamlit's file uploader
uploaded_image = st.file_uploader("Choose an image")

if uploaded_image is not None:
    file_path = save_upload_image(uploaded_image)
    
    if file_path:
        display_image = Image.open(file_path)

        # Extract features from the uploaded image
        features = extract_features(file_path, model, detector)
        
        if features is not None:
            index_pos = recommend(feature_list, features)

            # Absolute path for the output image
            output_image_path = os.path.join(current_dir, filenames[index_pos])
            output_image = cv2.imread(output_image_path)  # Read the output image using absolute path
            output_name = os.path.basename(filenames[index_pos])  # Get the image file name for display

            col1, col2 = st.columns(2)
            with col1:
                st.header("Your Image:")
                st.image(display_image, caption="Uploaded Image", use_column_width=True)
            with col2:
                st.header(output_name)
                st.image(output_image, caption="Matched Image", use_column_width=True)

            st.text(f"Extracted Features: {features}")
            st.text(f"Feature Shape: {features.shape}")
        else:
            st.text("Feature extraction failed.")
