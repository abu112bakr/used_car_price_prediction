# used_car_price_prediction
prediction of used car price using DecisionTreeRegressor, KNeighborsRegressor, SVM model and Linear Regression


Used Car Price Prediction with Machine Learning

This repository contains a machine learning project for predicting the price of used cars. The goal of this project is to develop models that can accurately estimate the selling price of used cars based on various features.
Dataset

The dataset used for this project is stored in a CSV file. Initially, the CSV file is loaded and data preprocessing steps are performed. The dataset had some missing values, including 18 null values in the "year" column, 23 null values in the "km_driven" column, and 4 null values in the "selling_price" column. These rows with missing values are dropped from the dataset to ensure data quality.
Data Preprocessing

The following preprocessing steps are applied to the dataset:

    Dropping Irrelevant Columns: The "name" column is dropped from the dataset as it is deemed irrelevant for predicting the car price.

    Label Encoding: Categorical features in the dataset are encoded using label encoding technique to convert them into numerical values. This allows the machine learning models to work with these features effectively.

    Train-Test Split: The preprocessed dataset is divided into training and testing sets. The data is split in such a way that 80% of the data is used for training the models, while the remaining 20% is used for evaluating the model's performance.

Machine Learning Models

The following machine learning models are used for predicting used car prices:

    Decision Tree Regressor: This model utilizes a decision tree algorithm to predict the selling price of used cars. The accuracy of the Decision Tree model is reported as 0.988.

    KNeighbors Regressor: The KNeighbors Regressor model predicts the car price based on the values of its k-nearest neighbors in the training data. The accuracy of the KNN model is reported as 0.983.

    Support Vector Machine (SVM) Model: SVM is a powerful algorithm used for both classification and regression tasks. In this project, an SVM model is employed for predicting car prices. The reported accuracy for the SVM model is -0.077.

    Linear Regression: Linear Regression is a popular algorithm for predicting numeric values. In this project, a Linear Regression model is utilized for predicting used car prices. The accuracy of the Linear Regression model is reported as 0.485.

Model Comparison

The models' performance is evaluated based on their R-squared values, which indicate how well the models fit the data. The R-squared values for the models are as follows:

    Linear Regression: 0.4853050345484957
    Decision Tree: 0.9883354783880333
    KNN: 0.9834886474267401
    SVM: -0.0770170343861023

Based on the R-squared values, it can be observed that the Decision Tree and KNN models perform significantly better than the Linear Regression and SVM models.

Please refer to the project files and code for more detailed implementation and analysis.
Requirements

The following libraries are required to run the code in this repository:

    scikit-learn
    pandas
    numpy

Please ensure that these dependencies are installed before running the code.
Usage

    Clone this repository to your local machine.
    Install the required dependencies mentioned in the requirements.txt file.
    Run the code in your preferred Python environment.
    Explore the implementation details and experiment with different models and parameters.

Feel free to provide any feedback or suggestions for further improvement. Happy coding!
