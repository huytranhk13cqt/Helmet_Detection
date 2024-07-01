# Helmet_Detection

This is an Object Detection app built with Streamlit, OpenCV, and the YOLOv10 model. The app allows users to upload an image, choose a pre-trained model, and adjust parameters such as confidence threshold and image size for detection. The processed image with detected objects will be displayed and can be downloaded.

# Setup For Using

1. Create new folder
2. Clone `yolov10` repo and install `requirements`

   ```bash
   git clone https://github.com/THU-MIG/yolov10
   cd yolov10
   pip install .
   ```

# Setup For Training

3. Download `dataset` and `pretrained_model` from [pretrained_model](https://www.google.com/url?q=https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt&sa=D&source=apps-viewer-frontend&ust=1719945093875421&usg=AOvVaw3ycv3Bjd0FfVxjIuxLstyg&hl=en) and [dataset](https://drive.google.com/file/d/1twdtZEfcw4ghSZIiPDypJurZnNXzMO7R/view)
4. Check CUDA version

   ```bash
   nvcc --version
   ```

5. Reinstall CUDA 11.7 from [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive)
6. Modify `datasets_dir` in `settings.yaml` at `C:\Users\your_user_name\AppData\Roaming\yolov10`

# Folder Structures

- `code`: source code
  - `predicts`: the location to save results during app execution
  - `test_results`: previously produced results
  - `trained_models`: 2 models trained by Google Colab and RTX 4050 Ti
  - `deploy.py`: main app using Streamlit
- `safety_helmet_dataset`: datasets for training and testing the models
- `yolov10`: YOLOv10 implementation main directory
- `training.py`: training sample on Local
- `Project.ipynb`: training sample on Google Colab

# Usage

- Once the environment is configured and dependencies are installed, you can begin using the YOLO model to detect objects in your images.
- To use this app, run this command interminal:

  ```bash
  streamlit run deploy.py
  ```

- Sample Image Predictions

  ![sample1](image.png)
  ![sample2](image-1.png)

# Recognitions

This project utilizes the YOLOv10 model from [THU-MIG](https://github.com/THU-MIG/yolov10)
