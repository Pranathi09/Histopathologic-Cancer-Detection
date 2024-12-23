import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import Network  # Update with the actual filename

# Neural Network Predefined Parameters
params_model = {
    "shape_in": (3, 46, 46),
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2
}

# Create instantiation of Network class
cnn_model = Network(params_model)

# Define computation hardware approach (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = cnn_model.to(device)  # Move model to correct device
model.load_state_dict(torch.load('weights.pt'))
model.eval()  # Set the model to evaluation mode

# Define colors
cream_color = "#FFF8E1"
light_brown_color = "#D2B48C"

# Page configuration
st.set_page_config(
    page_title="Cancer Detection App",
    page_icon=":microscope:",
)

# Apply custom CSS for styling
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background-color: {cream_color};
        }}
        .sidebar {{
            background-color: {light_brown_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# About page
st.title("Histopathologic Cancer Detection")
st.markdown(
    "A Streamlit WebApp for Histopathologic Cancer Detection using a Trained Neural Network."
)

# Demo page
def demo():
    st.title("Demo")
    st.markdown(
        "Upload an image to predict if the tissue is malignant or not. Click on 'Predict' to see the result."
    )
    col1, col2 = st.columns(2)
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif"])

    if uploaded_file is not None:
        # Preprocess the input image
        def preprocess_image(image):
            transform = transforms.Compose([
                transforms.Resize((46, 46)),
                transforms.ToTensor(),
            ])
            image = Image.open(image).convert("RGB")  # Ensure image is in RGB format
            image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to correct device
            return image

        def predict(image):
            with torch.no_grad():
                model_output = model(image)
                predicted_class = torch.argmax(model_output, dim=1).item()
                probability = torch.exp(model_output[:, 1]).item()
            return predicted_class, probability

        # Perform inference
        if st.button("Predict"):
            input_image = preprocess_image(uploaded_file)
            predicted_class, probability = predict(input_image)

            # Display the result
            classes = ['negative', 'tumor']
            output_image = Image.open(uploaded_file)
            col1.image(output_image, caption="Output Image", width=200)
            result_text = f'This is a {classes[predicted_class]} tissue image with a probability of {probability:.2%}.'
            col2.success(result_text)
            

# Main app
#if st.sidebar.button("Upload"):
demo()
st.sidebar.title("About")
st.sidebar.info("""
    This "Histopathologic Cancer Detection" WebApp uses advanced technology to analyze histopathologic images 
    and detect tumors. Users upload scans to Streamlit, and the system processes them with a smart algorithm to 
    provide an accurate output. This assists doctors in spotting issues faster, leading to better patient care 
    and healthier outcomes.
""")



