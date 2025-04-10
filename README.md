# Student Performance Prediction

## Introduction

This synthetic dataset models student performance based on factors such as study habits, sleep patterns, socioeconomic background, and class attendance. Each row represents a hypothetical student, with input features and a calculated grade. The dataset supports predictive modeling, exploratory analysis, and beginner-friendly machine learning workflows.

You can access the dataset [here](https://www.kaggle.com/datasets/stealthtechnologies/predict-student-performance-dataset/data).

## Dataset

The dataset includes the following features:
- **Socioeconomic Score**
- **Study Hours**
- **Sleep Hours**
- **Attendance (%)**

The target variable is:
- **Grade Category (0 to 4)**: Binned continuous grades into 5 equal-sized quintiles.

## Process

### Data Preprocessing
1. **Import Data**: Loading the dataset into the analysis environment.
2. **Normalization**: Scaling the features to bring them into the same range for machine learning models.

### Training Model
1. **Resampling**: Balancing the dataset through random oversampling of the minority class.
2. **Tuning and Cross-Validation**: Hyperparameter tuning using GridSearchCV with 5-fold cross-validation on 5 models.
   - Decision Tree
   - Logistic Regression
   - SVM
   - Naive Bayes
   - KNN

### Model Performance
- Metrics:
  - Precision
  - Recall
  - F1-Score
  - Accuracy
  - Weighted F1-Score
  - Confusion Matrix

### Testing Model
1. **Testing Models**: Evaluating performance on the same models (Decision Tree, Logistic Regression, SVM, Naive Bayes, KNN).
2. **Focus on F1-Score**: Given the imbalanced nature of the dataset, the weighted F1-score is the primary evaluation metric.

## Models Used
- **Decision Tree**
- **Logistic Regression**
- **SVM**
- **Naive Bayes**
- **KNN**

## Metrics Used
- **Precision**
- **Recall**
- **F1-Score**
- **Accuracy**
- **Weighted F1-Score**
- **Confusion Matrix**

## Training Process

### Grades Binning
Dividing the continuous 'Grades' data into 5 equal-sized quintiles (0, 1, 2, 3, 4), with the results stored in the 'Category' column.

### Feature and Target
- **Features**: Socioeconomic Score, Study Hours, Sleep Hours, Attendance (%)
- **Target**: Category (0-4)

### Random Oversampling
A technique to balance the dataset by increasing the number of samples in the minority class, with `random_state=42`.

### Models and Metrics Performance
The focus is mainly on the weighted F1-score, as accuracy alone is insufficient for imbalanced datasets.

## Tuning Hyperparameters with Cross-Validation

### Best Hyperparameters Found:

- **Decision Tree**:
  - criterion = gini
  - min_samples_split = 2
  - min_samples_leaf = 1

- **Logistic Regression**:
  - C = 10
  - penalty = l2
  - solver = saga
  - max_iter = 100

- **SVM**:
  - C = 100
  - gamma = 1
  - kernel = rbf

- **Naive Bayes**:
  - var_smoothing = 1e-09

- **KNN**:
  - n_neighbors = 3
  - metric = manhattan
  - weights = distance

## Testing Process

### Grades Binning
Dividing the continuous 'Grades' data into 5 equal-sized quintiles (0, 1, 2, 3, 4), stored in the 'Category' column.

### Feature and Target
- **Features**: Socioeconomic Score, Study Hours, Sleep Hours, Attendance (%)
- **Target**: Category (0-4)

### Original Imbalanced Dataset
- **Preserving Real-World Distribution**: Avoiding distortion of naturally imbalanced data.
- **Preventing Overfitting**: Oversampling could lead to memorization of synthetic samples.
- **Avoiding Information Loss**: Undersampling might result in valuable data being discarded.

### Models and Metrics Performance
The weighted F1-score is the key metric, as it provides a more balanced evaluation than accuracy alone in the context of imbalanced data.

## Conclusion

### The Best Model in Training and Testing

- **Training**:
  - **Model**: Decision Tree
  - **F1-Score**: 0.89190

- **Testing**:
  - **Model**: Decision Tree
  - **F1-Score**: 0.66289
