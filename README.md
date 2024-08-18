
# More Hyperparameter Tuning for Neural Networks Using Keras Tuner

This project demonstrates how to use Keras Tuner to optimize hyperparameters for a neural network model on a synthetic dataset. The dataset used is a nonlinear "circles" dataset, commonly used for binary classification tasks.

## Project Overview

This project involves the following steps:

1. **Installation of Dependencies:**
   - Ensure that Keras Tuner and other required libraries are installed.

2. **Data Import and Preprocessing:**
   - Load the nonlinear "circles" dataset.
   - Split the dataset into features (`X`) and target labels (`y`).
   - Further split the data into training and testing sets.
   - Scale the features using `StandardScaler` for improved model performance.

3. **Data Visualization:**
   - Plot the data to visualize the distribution of features and targets.

4. **Model Creation with Hyperparameter Tuning:**
   - Define a method `create_model` that allows Keras Tuner to tune various hyperparameters:
     - Activation function (e.g., ReLU, Tanh).
     - Number of neurons in the first layer.
     - Number of hidden layers and the number of neurons in each hidden layer.
   - Compile the model using binary cross-entropy as the loss function and the Adam optimizer.

5. **Running Keras Tuner:**
   - Utilize `Hyperband` tuner from Keras Tuner to search for the best hyperparameters.
   - The tuner performs a search over multiple epochs and configurations to identify the best models.

6. **Evaluation of Top Models:**
   - The top three models, as determined by Keras Tuner, are evaluated on the test dataset.
   - The accuracy and loss of these models are reported.

## Installation

To run this project, ensure you have Python and the following libraries installed:

```bash
pip install keras-tuner pandas scikit-learn tensorflow matplotlib
```

## How to Run the Project

1. **Install Dependencies** as outlined in the Installation section.
2. **Load and Preprocess Data:**
   - Load the dataset from the provided URL.
   - Split the data into training and testing sets, and scale the features.
3. **Define the Model Creation Function**:
   - Implement `create_model` with hyperparameter options for Keras Tuner to optimize.
4. **Run Keras Tuner** to search for the best model configuration.
5. **Evaluate Top Models**:
   - Evaluate the best models found by Keras Tuner on the test dataset.

## Results

- **Top Hyperparameters from the Tuning Process:**
  1. **Model 1:**
     - Activation: 'relu'
     - First layer units: 21
     - Number of hidden layers: 5
     - Hidden layers units: [11, 6, 21, 16, 21]
     - Accuracy: 97.60%
     - Loss: 0.1133
  2. **Model 2:**
     - Activation: 'relu'
     - First layer units: 26
     - Number of hidden layers: 4
     - Hidden layers units: [6, 21, 16, 21]
     - Accuracy: 97.60%
     - Loss: 0.1171
  3. **Model 3:**
     - Activation: 'tanh'
     - First layer units: 26
     - Number of hidden layers: 5
     - Hidden layers units: [21, 11, 1, 21, 16]
     - Accuracy: 95.60%
     - Loss: 0.1483

## Requirements

- Python 3.10
- Keras Tuner 1.4.6
- TensorFlow 2.14.0
- scikit-learn 1.0.2
- pandas 1.3.3
- matplotlib 3.4.3

## Conclusion

This project highlights how Keras Tuner can be used to efficiently search for optimal hyperparameters in neural network models. The use of Keras Tuner helped identify models with high accuracy for a challenging nonlinear dataset.

## Acknowledgments

Special thanks to the developers of Keras Tuner, TensorFlow, and the broader machine learning community for providing robust tools and resources for model development and optimization.
