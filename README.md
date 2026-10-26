# ASL Sign Language Classification

MACHINE LEARNING | IMAGE CLASSIFICATION | PYTHON ‧ SCIKIT-LEARN

A machine learning project that classifies American Sign Language (ASL) alphabet hand images using image preprocessing, Gray-Level Co-occurrence Matrix (GLCM) feature extraction, and traditional machine learning algorithms. The project compares K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest.

## 01. Project Overview

### Project Description

ASL Sign Language Classification is an image classification project developed to recognize static hand gestures representing letters of the American Sign Language alphabet. The project uses conventional machine learning rather than deep learning by transforming images into numerical texture features before classification.

The workflow consists of image preprocessing, GLCM feature extraction, feature selection, data normalization, model training, and performance evaluation.

### Objectives

- Build a classification model for static ASL alphabet hand images.
- Apply image preprocessing techniques to improve image representation.
- Extract texture features using GLCM.
- Compare KNN, SVM, and Random Forest classifiers.
- Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.

### Problem Statement

Communication barriers can arise when people do not understand sign language. Image-based sign classification offers a potential way to support sign language learning and accessibility.

This project explores whether visual texture features extracted from hand images can distinguish static ASL alphabet gestures using conventional machine learning algorithms.

### Dataset

The dataset was obtained from Kaggle:

[American Sign Language Dataset — Kaggle](https://www.kaggle.com/datasets/ayuraj/asl-dataset)

The original dataset contains 2,520 JPEG images across 36 categories: 26 alphabet letters and 10 digits. Each original category contains 70 images at a resolution of 400 × 400 pixels.

For this project, only static alphabet letters are included. The letters **J and Z** are excluded because their ASL representations involve movement, and the project focuses on static images. Numeric classes from 0 to 9 are also excluded.

The resulting classification task contains **24 alphabet classes**:

`A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y`

The dataset was expanded to 150 images per class through manual copying, as described in the project methodology, and images were resized to 100 × 100 pixels for processing.

### Solution and Workflow

The project implements the following pipeline:

1. **Data Understanding:** Inspect the dataset and visualize the number of images in each class.
2. **Image Preprocessing:** Apply grayscale conversion, brightness adjustment, sharpening, median filtering, Sobel edge detection, and min-max intensity normalization.
3. **Feature Extraction:** Calculate GLCM texture features from the processed images.
4. **Feature Selection:** Identify highly correlated features and remove redundant columns using a correlation threshold of 0.90.
5. **Data Splitting:** Split the feature dataset into training and testing subsets using an 80:20 ratio.
6. **Normalization:** Apply `StandardScaler`, fitting the scaler on training data and transforming the test data.
7. **Model Training:** Train KNN, SVM with an RBF kernel, and Random Forest.
8. **Evaluation:** Compare classification metrics and inspect confusion matrices.

### GLCM Features

The project extracts seven texture features:

- **Contrast:** Measures local intensity variation.
- **Dissimilarity:** Measures differences between neighboring pixel intensities.
- **Homogeneity:** Measures the concentration of GLCM values near the main diagonal.
- **Energy:** Measures texture uniformity.
- **Correlation:** Measures statistical relationships between neighboring pixel intensities.
- **Entropy:** Measures texture complexity and randomness.
- **Angular Second Moment (ASM):** Measures the uniformity of the gray-level co-occurrence distribution.

GLCM is calculated using distances of 1–5 pixels and four angles: 0°, 45°, 90°, and 135°.

## 02. Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Image Processing | OpenCV |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Development Environment | Jupyter Notebook |

### Machine Learning Models

| Model | Configuration |
|---|---|
| K-Nearest Neighbors (KNN) | `n_neighbors=5` |
| Support Vector Machine (SVM) | RBF kernel |
| Random Forest | 100 estimators |
| Data Split | 80% training, 20% testing |
| Feature Scaling | StandardScaler |
| Random State | 42 for the data split and configured models |

## 03. Screenshots / Results

### Machine Learning Workflow

The notebook includes visualizations for dataset distribution, feature correlation matrices before and after feature selection, and confusion matrices for the three classifiers.

- Dataset distribution across the 24 alphabet classes.
  <img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/ec517069-67df-41a9-9924-1a9555522583" />

- Example images.
  <img width="100" height="100" alt="hand_y (7)" src="https://github.com/user-attachments/assets/c0de644b-d051-4742-98d5-e50040f12279" />

  <img width="100" height="100" alt="hand_h (9)" src="https://github.com/user-attachments/assets/c6cd874c-c25b-4ad9-abaa-9d483e730d0e" />

- GLCM feature correlation heatmaps.
  <img width="1565" height="1408" alt="image" src="https://github.com/user-attachments/assets/f0b5e7c7-cbb8-47d5-9fd0-12e2890534ff" />

- Model performance comparison.
  <img width="399" height="167" alt="image" src="https://github.com/user-attachments/assets/568c3a86-7754-4c51-a2cc-f096e1ba303c" />

- Confusion matrices for KNN, SVM, and Random Forest.
  <img width="792" height="2395" alt="image" src="https://github.com/user-attachments/assets/80576a8f-c742-4d5c-94e9-b6fbde40e4c7" />


### Model Evaluation Results

The following metrics were obtained from the provided notebook output.

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
|---|---:|---:|---:|---:|
| KNN | 75.42% | 76.02% | 75.42% | 75.55% |
| SVM | 57.75% | 57.33% | 57.75% | 56.88% |
| Random Forest | **85.03%** | **85.19%** | **85.03%** | **85.02%** |

*Metrics are rounded to two decimal places. Precision, recall, and F1-score use the weighted averages reported by the notebook.*

### Key Insights

- **Random Forest performed best**, achieving 85.03% accuracy and an 85.02% weighted F1-score.
- KNN achieved 75.42% accuracy, indicating moderate classification performance using the extracted GLCM features.
- SVM achieved 57.75% accuracy, lower than the other two models under the current configuration.
- Random Forest provided the strongest overall balance of precision and recall among the evaluated models.
- The results suggest that GLCM-based texture features can provide useful information for static ASL alphabet classification, although performance varies by classifier.

### Usefulness

This project demonstrates an alternative to deep learning for image classification. By extracting handcrafted features from images, the classification process can operate on numerical feature vectors instead of directly training a neural network on raw pixels.

The model comparison helps identify which traditional machine learning algorithm performs best under the selected preprocessing, feature extraction, and training configuration.

### Practical Application

Potential applications include:

- Educational demonstrations for understanding static sign language classification.
- Prototypes for image-based ASL alphabet recognition.
- Benchmarking traditional machine learning approaches for hand gesture image classification.
- Further research into lightweight sign language recognition systems.

The current implementation focuses on static alphabet images. It does not recognize continuous signing, moving gestures, full sentences, or ASL grammar, and it should not be interpreted as a complete sign language translation system.

## 04. Installation & Usage

### Prerequisites

- Python 3.9 or a compatible Python version.
- Jupyter Notebook or JupyterLab.
- The ASL dataset downloaded from Kaggle.
- The dataset organized into alphabet-specific folders.

### Project Structure

```text
ml-ASLSignLanguageClassification/
├── project.ipynb
├── asl.zip/ <-- extract this
│   ├── a/
│   │   ├── hand_a (1).jpeg
│   │   ├── hand_a (2).jpeg
│   │   └── ...
│   ├── b/
│   ├── c/
│   ├── ...
│   ├── i/
│   ├── k/
│   ├── ...
│   └── y/
└── README.md
```

The `asl/` directory contains the 24 selected alphabet classes, excluding `j` and `z`. Each class folder contains the corresponding hand images.

Replace `notebook.ipynb` with the actual notebook filename in your repository.

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ml-ASLSignLanguageClassification.git
cd ml-ASLSignLanguageClassification
```

Install the required libraries:

```bash
pip install opencv-python numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the Project

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ayuraj/asl-dataset).
2. Select the 24 alphabet classes, excluding J and Z.
3. Organize the images into the corresponding folders inside `asl/`.
4. Apply the dataset expansion and resizing procedure used in the project.
5. Open the notebook in Jupyter Notebook or JupyterLab.
6. Execute the cells in order to inspect the dataset, preprocess images, extract GLCM features, and select features.
7. Run the data splitting and normalization steps.
8. Train KNN, SVM, and Random Forest.
9. Review the classification metrics and confusion matrices.

### Notes

- The notebook expects the dataset directory to be named `asl` relative to the notebook's working directory.
- The reported metrics reflect the provided notebook output and current model configurations.
- Manual copying increases the number of files but does not create new visual variation. If copied images are duplicates, ensure identical or near-identical images do not appear across both training and test sets, as this can inflate evaluation results.
- The project uses handcrafted GLCM features and conventional machine learning algorithms rather than a CNN-based classifier.
