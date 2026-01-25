# Health Risk Score Prediction using Multiple Linear Regression

## Project Overview
This project builds an end-to-end machine learning pipeline to predict a continuous
health risk score using demographic and lifestyle indicators. The model is trained
using Multiple Linear Regression and evaluated with standard regression metrics.

## Dataset
- Source: Public health survey dataset
- Samples: 253,680
- Features include:
  - Age
  - BMI
  - Smoking status
  - Blood pressure
  - Physical activity
  - Mental & physical health indicators

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Workflow
1. Data loading and cleaning
2. Feature selection
3. Feature scaling
4. Train-test split
5. Model training
6. Baseline comparison
7. Model evaluation (RMSE, R²)
8. Visualization and residual analysis

## Results
- R² Score: ~0.81
- Model significantly outperforms baseline predictor
- Residuals show near-normal distribution indicating good fit

## Visualizations
- Actual vs Predicted Health Risk
- Residual Distribution
- Residuals vs Predictions

## Conclusion
The model demonstrates strong predictive capability and provides interpretable
feature importance insights, making it suitable for real-world health analytics.

## Future Improvements
- Logistic Regression for disease prediction
- Regularization (Ridge, Lasso)
- Cross-validation
