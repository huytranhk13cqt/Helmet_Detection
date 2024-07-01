import logging

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLOv10

# Configure logging
logging.basicConfig(level=logging.INFO)

# Declare the model paths
MODEL_PATHS = {
    "best_colab": "./trained_models/best_colab.pt",
    "best_self": "./trained_models/best_self.pt"
}
SAVE_PATH = "./predicts/output.png"


def process_image(image, conf_threshold, img_size, model_path):
    """Process the image using YOLOv10 model."""
    try:
        model = YOLOv10(model_path)
        model.conf = conf_threshold
        model.imgsz = img_size
        result = model(image)[0]
        # Ensure the image is saved in RGB format
        result_image = result.plot()  # Assuming plot() returns an RGB image
        rgb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(SAVE_PATH, rgb_image)
        logging.info("Image processed successfully.")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise


def main():
    """Main function to run the Streamlit app."""
    st.title('Object Detection App')
    model_choice = st.selectbox('Choose a model', list(MODEL_PATHS.keys()))

    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    conf_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.5)
    img_size = st.slider('Image Size', 256, 1024, 640, step=32)

    if file is not None:
        st.image(file, caption="Uploaded Image")
        image = Image.open(file)
        image = np.array(image)

        try:
            process_image(image, conf_threshold, img_size,
                          MODEL_PATHS[model_choice])
            predict_image = cv2.imread(SAVE_PATH)
            predict_image_rgb = cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB)
            st.image(predict_image_rgb, caption="Predict Image")
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
