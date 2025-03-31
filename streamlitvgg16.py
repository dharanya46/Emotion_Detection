import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load the trained VGG16 model
def load_model():
    # Define the model architecture (same as during training)
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 7)  # 7 classes for emotions
    model.classifier.add_module('dropout', nn.Dropout(0.5))  # Add dropout layer
    model.load_state_dict(torch.load('vgg16model.pth', map_location=torch.device('cpu')))  # Load the saved weights
    model.eval()  # Set to evaluation mode
    return model

# Define transformations for the uploaded image
def transform_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB format
        transforms.Resize((224, 224)),  # Resize to 224x224 for VGG16
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to predict emotion
def predict_emotion(image, model):
    # Transform the image
    image_tensor = transform_image(image)
    
    # Predict the emotion
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        
    # Define the emotion classes (change these to your actual emotion classes)
    emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']
    
    return emotions[predicted_class.item()]

# Streamlit UI
def main():
    st.title("Emotion Detection from Image")
    
    # Load the model once
    model = load_model()

    st.write("Upload an image to predict the emotion")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Open the image using PIL
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict the emotion
        if st.button('Predict Emotion'):
            # Get the emotion prediction
            predicted_emotion = predict_emotion(image, model)
            
            # Display the prediction
            st.write(f"Predicted Emotion: {predicted_emotion}")
        
if __name__ == "__main__":
    main()
