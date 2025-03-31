import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


def load_model():
   
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 7) 
    model.classifier.add_module('dropout', nn.Dropout(0.5)) 
    model.load_state_dict(torch.load('vgg16model.pth', map_location=torch.device('cpu')))  
    model.eval()  
    return model


def transform_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    return transform(image).unsqueeze(0) 


def predict_emotion(image, model):
   
    image_tensor = transform_image(image)
    
   
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        
    
    emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']
    
    return emotions[predicted_class.item()]


def main():
    st.title("Emotion Detection from Image")
    
   
    model = load_model()

    st.write("Upload an image to predict the emotion")

   
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
       
        image = Image.open(uploaded_file)
        
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        
        if st.button('Predict Emotion'):
            
            predicted_emotion = predict_emotion(image, model)
            
            
            st.write(f"Predicted Emotion: {predicted_emotion}")
        
if __name__ == "__main__":
    main()
