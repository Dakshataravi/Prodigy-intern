# Linear Regression Project (House Price Prediction)

## 📌 Description
This project predicts house prices based on:
- Square Footage
- Number of Bedrooms
- Number of Bathrooms

## 📂 Files
- main.py → Train model
- predict.py → Predict using trained model
- data.csv → Dataset (you must download from Kaggle)
- model.pkl → Saved model

## 🚀 Steps to Run

1. Install dependencies:
pip install pandas scikit-learn

2. Add dataset:
Download from:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Rename train.csv → data.csv

3. Train model:
python main.py

4. Predict:
python predict.py

## ⚠️ Note
Make sure column names match:
GrLivArea, BedroomAbvGr, FullBath, SalePrice
