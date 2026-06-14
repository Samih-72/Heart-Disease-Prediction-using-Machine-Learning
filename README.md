❤️ Heart Disease Analysis & Prediction
Project Overview

This project focuses on analyzing heart disease data to identify key health indicators associated with cardiovascular risk. The workflow includes data cleaning, exploratory data analysis (EDA), feature engineering, feature scaling, and predictive analytics preparation.

Dataset Features
Age
Sex
Chest Pain Type
Resting Blood Pressure
Cholesterol
Fasting Blood Sugar
Maximum Heart Rate
Exercise Angina
Oldpeak
ST Slope
Heart Disease (Target Variable)
Project Workflow
1. Data Cleaning
Checked for missing values
Identified invalid values
Replaced zero values in Cholesterol and RestingBP with meaningful averages
2. Exploratory Data Analysis (EDA)
Distribution analysis using histograms
Heart disease class distribution
Count plots for categorical variables
Boxplots and violin plots
Correlation heatmap
3. Feature Engineering

Created additional features such as:

AgeGroup
HighCholesterol
LowMaxHR
Age_Cholesterol
Age_BP
RiskScore
OldpeakRisk
4. Feature Scaling

Applied StandardScaler to normalize numerical features.

5. Correlation Analysis

Identified factors most associated with heart disease risk.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Key Insights
Age and cardiovascular indicators strongly influence heart disease risk.
Cholesterol and blood pressure abnormalities contribute significantly to risk assessment.
Engineered features improve understanding of patient risk profiles.
Future Improvements
Train machine learning models for prediction.
Hyperparameter tuning.
Model deployment using Flask/Streamlit.
