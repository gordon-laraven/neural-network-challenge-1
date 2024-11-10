# Student Loan Risk Prediction with Deep Learning

## Table of Contents
1. [Background](#background)
2. [Instructions](#instructions)
   - [Part 1: Prepare the Data for Use on a Neural Network Model](#part-1-prepare-the-data-for-use-on-a-neural-network-model)
   - [Part 2: Compile and Evaluate a Model Using a Neural Network](#part-2-compile-and-evaluate-a-model-using-a-neural-network)
   - [Part 3: Predict Loan Repayment Success by Using Your Neural Network Model](#part-3-predict-loan-repayment-success-by-using-your-neural-network-model)
   - [Part 4: Discuss Creating a Recommendation System for Student Loans](#part-4-discuss-creating-a-recommendation-system-for-student-loans)
3. [References](#references)

## Background
In an effort to improve the accuracy of interest rates offered to borrowers, our company is focusing on developing a model to predict the likelihood that a borrower will be able to repay their student loans. Utilizing a dataset containing information from previous student loan recipients, including credit ranking and other relevant features, we will create a model capable of forecasting loan repayment success.

## Instructions
The project is divided into several parts. The following sections illustrate the steps required to prepare the data, compile and evaluate the model, conduct predictions, and discuss the recommendation system's creation.

### Part 1: Prepare the Data for Use on a Neural Network Model
1. **Read the Dataset**: Import the dataset using Pandas.
    ```python
    import pandas as pd
    file_path = "https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv"
    loans_df = pd.read_csv(file_path)
    ```
   
2. **Create Features and Target Datasets**: Define the features (`X`) and target (`y`) datasets.
    ```python
    y = loans_df["credit_ranking"]
    X = loans_df.drop(columns=["credit_ranking"])
    ```

3. **Split the Data**: Utilize `train_test_split` to create training and testing datasets.
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    ```

4. **Scale the Features**: Apply `StandardScaler` to scale the feature data.
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

### Part 2: Compile and Evaluate a Model Using a Neural Network
1. **Create the Model**: Define a deep neural network using TensorFlow.
    ```python
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    n_inputs = X_train_scaled.shape[1]
    model = Sequential()
    model.add(Dense(units=10, activation='relu', input_dim=n_inputs))
    model.add(Dense(units=5, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    ```

2. **Compile the Model**: Set the loss function, optimizer, and metrics.
    ```python
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ```

3. **Train the Model**: Fit the model with the training data.
    ```python
    model.fit(X_train_scaled, y_train, epochs=50, verbose=1)
    ```

4. **Evaluate the Model**: Assess the model's performance using the testing data.
    ```python
    model.evaluate(X_test_scaled, y_test)
    ```

5. **Save the Model**: Export the trained model for future predictions.
    ```python
    model.save('student_loans.keras')
    ```

### Part 3: Predict Loan Repayment Success by Using Your Neural Network Model
1. **Reload the Model**: Load the previously saved model.
    ```python
    loaded_model = tf.keras.models.load_model('student_loans.keras')
    ```

2. **Make Predictions**: Use the model to predict outcomes on testing data.
    ```python
    predictions = loaded_model.predict(X_test_scaled)
    predictions_df = pd.DataFrame(predictions, columns=['Predicted_Credit_Ranking'])
    predictions_df['Predicted_Credit_Ranking'] = round(predictions_df['Predicted_Credit_Ranking'], 0)
    ```

3. **Classification Report**: Generate a report comparing predicted and actual values.
    ```python
    from sklearn.metrics import classification_report
    print(classification_report(y_test, predictions_df['Predicted_Credit_Ranking']))
    ```

### Part 4: Discuss Creating a Recommendation System for Student Loans
1. **Data Collection**: Identify data needed for a robust recommendation system including student demographics, financial information, and loan product details.
   
2. **Filtering Method**: Determine if content-based filtering, collaborative filtering, or context-based filtering will best suit the dataset selected. For instance, using characteristics of borrowers and loan products aligns well with content-based filtering.

3. **Challenges**: Consider potential challenges such as data privacy, requirement for real-time data accuracy, and the evolving nature of financial products impacting recommendations.

## References
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pathlib Documentation](https://docs.python.org/3/library/pathlib.html)

