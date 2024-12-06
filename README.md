# Eye Disease Identification using DenseNet201

This project aims to identify different eye diseases using a pre-trained DenseNet201 model. The dataset used contains images of eyes classified into four categories: cataract, diabetic retinopathy, glaucoma, and normal.

## Project Overview

The project follows these main steps:
1. **Data Collection and Preparation**:
    - Download the dataset from Kaggle.
    - Extract and preprocess the images.
2. **Data Preprocessing**:
    - Resize images to 128x128 pixels.
    - Split the data into training and testing sets.
    - Normalize the image data.
3. **Model Training**:
    - Build a neural network using DenseNet201 as the base model.
    - Train the model with the preprocessed data.
4. **Model Evaluation**:
    - Evaluate the model's performance on the test set.
    - Plot loss and accuracy graphs.

## Dependencies

The project requires the following dependencies:
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- TensorFlow
- Keras
- OpenCV
- Pillow

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/DarkLord-13/Machine-Learning-01.git
    ```

2. Navigate to the project directory:
    ```sh
    cd Machine-Learning-01
    ```

3. Install the required packages:
    ```sh
    pip install pandas numpy matplotlib seaborn tensorflow keras opencv-python pillow kaggle
    ```

4. Download the dataset from Kaggle:
    ```sh
    kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification
    ```

5. Extract the dataset:
    ```sh
    unzip eye-diseases-classification.zip -d dataset
    ```

6. Open the Jupyter Notebook `EyeDiseaseIdentification(DenseNet201).ipynb` and run the cells to execute the project steps:
    ```sh
    jupyter notebook EyeDiseaseIdentification(DenseNet201).ipynb
    ```

## Usage

1. **Data Collection and Preparation**:
    - Download and extract the dataset using Kaggle API.

2. **Data Preprocessing**:
    - Resize images to 128x128 pixels.
    - Split the dataset into training and testing sets.
    - Normalize the image data.

3. **Model Training**:
    - Build a neural network using DenseNet201 as the base model.
    - Train the model with the preprocessed data.

4. **Model Evaluation**:
    - Evaluate the model's performance on the test set.
    - Plot loss and accuracy graphs to visualize the model's performance.

## Results

The trained DenseNet201 model achieved approximately 90% accuracy on the test set. The model's performance can be visualized using the loss and accuracy plots generated during training.

## License

This project is licensed under the MIT License.
