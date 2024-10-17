import numpy as np
import pickle
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the feature list and filenames from pickle files
feature_list = np.array(pickle.load(open("./embedding.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

# Initialize the ResNet50 model and MTCNN detector
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")
detector = MTCNN()

# Streamlit app title and description
st.title("Face Similarity Detector")
st.write("Upload an image to detect faces and find the most similar one.")

# Upload the sample image using Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    sample_img = cv2.imdecode(file_bytes, 1)

    # Detect faces in the image
    results = detector.detect_faces(sample_img)

    # Check if at least one face is detected
    if results:
        # Get the bounding box of the first detected face
        x, y, width, height = results[0]['box']
        
        # Extract the face from the image
        face = sample_img[y:y + height, x:x + width]

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

        # Calculate cosine similarity with all features in the feature list
        similarity = []
        for feature in feature_list:
            sim = cosine_similarity(result.reshape(1, -1), feature.reshape(1, -1))
            similarity.append(sim[0][0])  # Get the similarity score

        # Find the index of the most similar face
        index_pos = sorted(enumerate(similarity), reverse=True, key=lambda x: x[1])[0][0]

        # Load and display the most similar face
        output_image = cv2.imread(filenames[index_pos])
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        output_name = filenames[index_pos].split("\\")[1]

        st.image(output_image_rgb, caption=f"Most Similar Face: {output_name}", use_column_width=True)
        st.success(f"Found the most similar face: {output_name}")
    else:
        st.error("No face detected in the image.")
