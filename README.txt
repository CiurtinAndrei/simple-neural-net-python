Multi-Layer Perceptron (MLP) Classifier
This Python program implements a Multi-Layer Perceptron (MLP) classifier using the scikit-learn library, aimed at supervised learning tasks. It is adept at categorizing input data into one of seven possible categories, 
leveraging a comprehensive dataset split between training and testing subsets.

Dataset Details:

The dataset comprises two CSV files: one with 2310 samples, each featuring 19 distinct attributes, and another containing 2310 corresponding labels with values ranging from 1 to 7.
To foster robust model training and evaluation, the dataset is divided into training and testing sets, with 75% of the data used for training the model, and the remaining 25% reserved for testing its performance.
Model Architecture:

The MLP model features an architecture with two hidden layers, each consisting of 150 neurons. This configuration allows the model to learn complex patterns in the data, facilitating accurate predictions.

Training Parameters:

A learning rate of 0.01 is set for the training process, guiding the rate at which the model's weights are updated and thus optimizing performance over iterations.
Performance:

The MLP model showcases commendable efficacy, achieving an average accuracy of over 92% on the testing set, indicating its strong predictive capabilities across the board.

Usage Instructions:

Ensure the CSV files for the dataset and labels are correctly formatted and placed in the specified directory for the program to access.
Execute the Python script to initiate training of the MLP model with the provided dataset.
Upon completion of training, the model's performance can be assessed on the testing set using various metrics, including accuracy, to gauge its effectiveness.
Currently, this model has an average accuracy of about  0.94 (94 %), but the number of hidden layers, number of neurons per layer and the learning rate can be adjusted in order to obtain better results.
