# main.py

from fastapi import FastAPI
import uvicorn
import json
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()


if __name__ == "__main__" :
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")

class InputList(BaseModel):
    Hours_Studied: Optional[int]
    Attendance: Optional[int]
    Parental_Involvement: Optional[int]
    Access_to_Resources: Optional[int]
    Extracurricular_Activities: Optional[int]
    Sleep_Hours: Optional[int]
    Previous_Scores: Optional[int]
    Motivation_Level: Optional[int]
    Internet_Access: Optional[int]
    Tutoring_Sessions: Optional[int]
    Family_Income: Optional[int]
    Teacher_Quality: Optional[int]
    School_Type: Optional[int]
    Peer_Influence: Optional[int]
    Physical_Activity: Optional[int]
    Learning_Disabilities: Optional[int]
    Parental_Education_Level: Optional[int]
    Distance_from_Home: Optional[int]
    Gender: Optional[int]
    

@app.post("/predict")
def predict(x_input_list: InputList):
    
    def manual_correlation_matrix(df):
        # Get the list of columns
        columns = df.columns
        n = len(columns)

        # Initialize an empty matrix to store correlations
        corr_matrix = pd.DataFrame(index=columns, columns=columns)

        # Nested loop to calculate correlation between each pair of columns
        for i in range(n):
            for j in range(n):
                # Get the lists of values for the two columns
                x = df[columns[i]].tolist()
                y = df[columns[j]].tolist()

                # Calculate correlation using the manual_correlation function
                corr_matrix.iloc[i, j] = manual_correlation(x, y)

        return corr_matrix.astype(float)

    # The manual_correlation function (as before)
    def manual_correlation(x, y):
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        x_deviation = [xi - mean_x for xi in x]
        y_deviation = [yi - mean_y for yi in y]
        sum_product_deviation = sum(xi * yi for xi, yi in zip(x_deviation, y_deviation))
        sum_x_deviation_squared = sum(xi ** 2 for xi in x_deviation)
        sum_y_deviation_squared = sum(yi ** 2 for yi in y_deviation)
        correlation = sum_product_deviation / (sum_x_deviation_squared**0.5 * sum_y_deviation_squared**0.5)
        return correlation
    

    def predict_exam_score(X_input, theta):
      # Add intercept term (a 1 in the first position) to the input data
      X_input_b = np.insert(X_input, 0, 1)  # Insert 1 for the intercept
      # Predict the output by calculating dot product of theta and input
      y_pred = np.dot(X_input_b, theta)

      return y_pred
    
    def toList(df,input):
      input_list = []
      for topic, value in input.dict().items():
        if value is None:
          value = get_mode_or_mean(df[topic])
        else:
            try:
                # Convert to the correct data type
                value = int(value) if df[topic].dtype in ['int64', 'float64'] else value
            except ValueError:
                print(f"Invalid input for {topic}, using mode instead.")
                value = get_mode_or_mean(df[topic])

        input_list.append(int(value))
      return input_list      
                
            
    def get_mode_or_mean(column):
      """Function to return mode or mean depending on the data type."""
      if column.dtype in ['int64', 'float64']:  # If numeric, use mean
          return column.mean()
      else:  # If categorical, use mode
          return column.mode()[0]
    
    def provide_improvement_suggestions(df, input_list):
      # Step 1: Calculate the correlation matrix using your manual_correlation_matrix function
      correlation_matrix = manual_correlation_matrix(df)

      # Step 2: Extract correlation values with 'Exam_Score'
      score_correlations = correlation_matrix['Exam_Score'].drop('Exam_Score')  # Exclude self-correlation

      # Step 3: Define thresholds for strong correlation
      strong_positive_threshold = 0.05  # Adjust as per your needs
      strong_negative_threshold = -0.05

      # Initialize lists to store suggestions
      improve_factors = []
      reduce_factors = []

      # Step 4: Loop through the correlations and give feedback based on the input values
      for i, (feature, correlation_value) in enumerate(score_correlations.items()):
          user_input_value = input_list[i]  # Get the corresponding input for this feature

              # Non-binary variables
          if correlation_value > strong_positive_threshold:
              improve_factors.append(f"{feature}")
          elif correlation_value < strong_negative_threshold:
              reduce_factors.append(f"{feature}")

      suggestions = {
          "improve_factors": improve_factors,
          "reduce_factors": reduce_factors
      }
      return suggestions

    target_data_path = './studentPerformanceCleaned.csv'
    data = pd.read_csv(target_data_path)

    # Selecting independent variables (features) and dependent variable (target)
    X = data.drop(columns=['Exam_Score'])  # Features
    y = data['Exam_Score']  # Target variable (exam scores)
    

    X_b = np.c_[np.ones((len(X), 1)), X]  # Add the intercept column

    # Step 2: Compute X^T * X
    X_transpose_X = X_b.T.dot(X_b)

    # Step 3: Compute the inverse of (X^T * X)
    inverse_X_transpose_X = np.linalg.inv(X_transpose_X)

    # Step 4: Compute X^T * y
    X_transpose_y = X_b.T.dot(y)

    # Step 5: Compute the coefficients (theta)
    theta = inverse_X_transpose_X.dot(X_transpose_y)

    x_input_list = toList(data,x_input_list)
    predicted_exam_score = predict_exam_score(x_input_list, theta)
    suggestion = provide_improvement_suggestions(data, x_input_list)

    return {"predicted_exam_score": predicted_exam_score, "suggestion": suggestion}
