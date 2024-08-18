
# Sensor Data Classification Project

This is my Machine Learning Project for the Master Program at University of South Florida

This project is part of my Machine Learning and Deep Learning portfolio, showcasing my expertise in building and optimizing neural network models using advanced techniques. The goal of this project is to classify transportation modes based on accelerometer data collected from cell phones, a critical task in activity monitoring and context-aware services.

## Objective

The primary objective of this project is to accurately classify the mode of transportation (e.g., still, walking, bus, car, train) using time-series data from accelerometers. The project leverages deep learning techniques, including Neural Networks, Recurrent Neural Networks (RNNs), and Long Short-Term Memory (LSTM) models, to achieve high classification accuracy.

## Research Questions and Answers

**Q1. Can we accurately classify the mode of transportation using accelerometer data from cell phones?**


**Q2. How does the structure of neural networks, specifically LSTM and GRU layers, impact the classification accuracy?**


**Q3. How do different neural network models compare in terms of predictive accuracy for transportation mode classification?**


## Data Description

The dataset used in this project, `movement.csv`, contains accelerometer data across 20 time steps and a `Target` variable that represents the mode of transportation. The data is preprocessed to transform the `Target` variable into ordinal values, suitable for machine learning models.

## Methodology

### 1. Data Preprocessing

- **Ordinal Encoding:** The `Target` column was converted from categorical text to ordinal values using `OrdinalEncoder` to facilitate multiclass classification.
- **Data Reshaping:** The time-series data (columns 1 to 20) was reshaped into a 3D format to be compatible with Keras models, essential for neural networks and RNNs.

### 2. Model Development

#### Neural Network Using Keras

- **Architecture:** A simple neural network with one hidden layer was built using Keras to establish a baseline. This model achieved an accuracy of 58.33%.
- **Optimization:** Further, a deep neural network with two or more hidden layers was implemented, improving accuracy to 63.89%.

#### Recurrent Neural Network (RNN) with One Layer and Early Stopping

- **Architecture:** A simple RNN model was developed to capture sequential dependencies in the data. Early stopping was employed to prevent overfitting, achieving an accuracy of 61.11%.
- **Impact:** The use of early stopping ensured that the model did not overfit, resulting in better generalization to unseen data.

#### Deep LSTM Model (Two Layers)

- **Architecture:** A deep LSTM model with two layers was implemented to leverage the power of LSTMs in handling time-series data. This model outperformed all others, achieving the highest accuracy of 72.22%.
- **Significance:** The LSTM model's ability to capture long-term dependencies made it the best performer in this project, demonstrating the value of deep learning in complex classification tasks.

#### GRU Models

- **Architecture:** Both simple and deep GRU models were built to compare their performance with LSTM models. While effective, GRUs did not surpass LSTM models in accuracy.

### 3. Cross-Validation

- **10-Fold Cross Validation:** To ensure robustness, the best-performing model (deep LSTM) was subjected to 10-fold cross-validation. This process yielded a mean accuracy of 65.83%, demonstrating the model's stability across different data splits.

### 4. Model Evaluation

- **Accuracy and Loss Metrics:** Each model was evaluated using accuracy and loss metrics to determine its effectiveness. The deep LSTM model stood out with a test accuracy of 72.22% and a low loss of 0.71.

## Advanced Techniques and Libraries

This project utilized several advanced techniques and libraries to enhance the model-building process:

1. **Keras Sequential API:** For building and training neural networks, including LSTM and GRU models.
2. **TensorFlow:** Provided the computational backend for all neural network operations.
3. **EarlyStopping:** Used to prevent overfitting in recurrent models, ensuring better generalization.
4. **K-Fold Cross-Validation:** Applied to validate the model's performance across different data subsets.

## Key Features and Insights

### **Deep Learning and Sequential Models**

- **Deep LSTM Model:** The deep LSTM with two layers was the most effective, demonstrating the power of LSTMs in capturing long-term dependencies in time-series data.
- **GRU and RNN Models:** These models provided solid performance but did not surpass the LSTM in accuracy.

### **Cross-Validation Results**

- **10-Fold Cross-Validation:** The deep LSTM model showed a mean accuracy of 65.83% across folds, highlighting its consistency and robustness.

## Model Performance Summary

- **Baseline Model (DummyClassifier):**
  - **Train Accuracy:** 35%
  - **Test Accuracy:** 39%

- **Neural Network (1 Hidden Layer):**
  - **Loss:** 1.25
  - **Accuracy:** 58.33%

- **Deep Neural Network (2 or More Hidden Layers):**
  - **Loss:** 0.82
  - **Accuracy:** 63.89%

- **Simple RNN Model (1 Layer):**
  - **Loss:** 0.72
  - **Accuracy:** 61.11%

- **Deep LSTM Model (2 Layers):**
  - **Loss:** 0.71
  - **Accuracy:** 72.22%

- **Simple GRU Model (1 Layer):**
  - **Loss:** 0.71
  - **Accuracy:** 58.33%

- **Deep GRU Model (2 Layers):**
  - **Loss:** 0.72
  - **Accuracy:** 66.67%

## Research Questions and Answers

**Q1. Can we accurately classify the mode of transportation using accelerometer data from cell phones?**

**Answer:** Yes, by using advanced deep learning models such as LSTMs and GRUs, we can achieve high accuracy in classifying the mode of transportation. The best model in this project, a deep LSTM with two layers, achieved an accuracy of 72.22%, demonstrating the effectiveness of these techniques in handling time-series data.

**Q2. How does the structure of neural networks, specifically LSTM and GRU layers, impact the classification accuracy?**

**Answer:** The structure of the neural networks plays a critical role in classification accuracy. LSTM layers, known for their ability to capture long-term dependencies, provided superior performance compared to simpler neural networks and GRU models. The deep LSTM model with two layers outperformed other architectures, highlighting the importance of depth and recurrence in neural networks for this type of data.

**Q3. How do different neural network models compare in terms of predictive accuracy for transportation mode classification?**

**Answer:** The comparison of different models showed that the deep LSTM model with two layers was the most effective, achieving the highest accuracy of 72.22%. While other models, such as simple RNNs and deep GRUs, also performed well, the LSTM’s ability to model sequential data gave it an edge over the others.

## Best Model

The **Deep LSTM Model (2 Layers)** achieved the best performance with a test accuracy of 72.22%. This model was selected for its ability to generalize well to unseen data, making it the most reliable model for this classification task.

## Comparison to Baseline

The optimized Deep LSTM model significantly outperformed the baseline, achieving 72.22% accuracy compared to the baseline's 39%, showcasing a marked improvement in predictive accuracy.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for any enhancements or bug fixes.

## Acknowledgments

- **Akanksha Kushwaha** for project submission.
- **TensorFlow and Keras Documentation** for guidance on model implementation and evaluation.
