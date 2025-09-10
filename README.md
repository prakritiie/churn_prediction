# Customer Churn Prediction

### Overview
This project uses an **Artificial Neural Network (ANN)** to predict customer churn. By analyzing customer data, the model predict if a customer is likely to churn.

### Model Architecture
The model is a simple feedforward neural network built with **Keras** and **TensorFlow**.


* **Input Data:** The model takes in various customer features like tenure, monthly charges, and gender.
* **Internal Calculation:** The neural network processes these features through its layers.
* **Output Probability:** The final sigmoid activation layer outputs a single value between 0 and 1. This value represents the probability of the customer churning.

A value close to 1 means the model is highly confident the customer will churn.
A value close to 0 means the model believes the customer is very likely to stay.
