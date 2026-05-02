---
title: Podcast Listening Time Predictor
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
python_version: "3.10"
app_file: app.py
pinned: false
license: gpl-3.0
---

# Podcast Listening Time Prediction (Regression)

This project is part of my **"Become a Pro"** Data Science portfolio. It uses a machine learning approach to predict the total listening time of podcast episodes based on various metadata.

## 🚀 Project Overview
The goal of this competition was to build a regression model that minimizes **RMSE** while handling complex categorical data. I achieved this by implementing an **XGBoost** regressor and a custom preprocessing pipeline.

### 📊 Key Features
- **Problem Type:** Regression
- **Target Variable:** `Listening_Time_minutes`
- **Evaluation Metric:** RMSE
- **Insights:** The model revealed that Host Popularity and Episode Sentiment are the primary drivers of user engagement.

## 🛠️ Technical Stack
- **Language:** Python 3.10
- **Model:** XGBoost
- **Deployment:** Streamlit & Hugging Face Spaces
- **Libraries:** Pandas, Scikit-learn, Joblib, NumPy

## 📂 Files Included
- `app.py`: The Streamlit application interface.
- `xgb_podcast_model.pkl`: The trained XGBoost model.
- `features.pkl`: Saved feature list for column alignment.
- `requirements.txt`: Environment dependencies.

## 📝 How to run locally
1. Create a virtual environment with Python 3.10:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate