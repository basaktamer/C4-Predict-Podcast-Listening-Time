import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model and the scaler
try:
    # Ensure these .pkl files are in the same folder as app.py
    model = joblib.load('xgb_podcast_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    # Get the exact feature names the scaler expects
    model_features = scaler.feature_names_in_
except Exception as e:
    st.error(f"Error loading model files: {e}")

st.set_page_config(page_title="Podcast Predictor", page_icon="🎙️")

# Custom CSS for Professional UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #6c5ce7;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎙️ Podcast Listening Time Predictor")
st.info("Input the metadata below to predict the total listening minutes.")

# 2. User Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # NOTE: Names must match your training data capitalization exactly
        genre = st.selectbox("Genre", ["Tech", "Comedy", "True Crime", "History", "Society", "Business", "Health"])
        sentiment = st.selectbox("Episode Sentiment", ["Positive", "Neutral", "Negative"])
        host_pop = st.slider("Host Popularity (%)", 0, 100, 50)
        
    with col2:
        pub_day = st.selectbox("Publication Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        pub_time = st.selectbox("Publication Time", ["Morning", "Afternoon", "Evening", "Night"])
        guest_pop = st.slider("Guest Popularity (%)", 0, 100, 20)

    submit = st.form_submit_button("Predict Listening Time")

if submit:
    # 3. Create DataFrame
    input_data = pd.DataFrame({
        'Genre': [genre],
        'Episode_Sentiment': [sentiment],
        'Host_Popularity_percentage': [host_pop],
        'Guest_Popularity_percentage': [guest_pop],
        'Publication_Day': [pub_day],
        'Publication_Time': [pub_time]
    })

    # 4. Processing logic (Matching your Kaggle pipeline)
    sentiment_map = {'Negative': 1, 'Neutral': 2, 'Positive': 3}
    time_order = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
    
    input_data['Episode_Sentiment_Score'] = input_data['Episode_Sentiment'].map(sentiment_map)
    input_data['Publication_Time_Ordinal'] = input_data['Publication_Time'].map(time_order)
    input_data['is_weekend'] = input_data['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Drop raw string columns used for mapping to avoid duplicates
    input_prep = input_data.drop(['Episode_Sentiment', 'Publication_Time', 'Publication_Day'], axis=1)
    
    # 5. One-Hot Encoding and Alignment
    input_encoded = pd.get_dummies(input_prep)
    input_aligned = input_encoded.reindex(columns=model_features, fill_value=0)
    
    # 6. Scaling & Prediction
    try:
        input_scaled = scaler.transform(input_aligned)
        prediction = model.predict(input_scaled)[0]
        
        # Display the Result
        st.markdown("---")
        # Ensure prediction isn't negative due to model variance
        final_prediction = max(0, prediction)
        
        if final_prediction < 10:
            st.warning(f"### Predicted Listening Time: {final_prediction:.2f} minutes")
            st.write("💡 *Note: Low prediction suggests this metadata matches short-form content patterns.*")
        else:
            st.success(f"### Predicted Listening Time: {final_prediction:.2f} minutes")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.caption("Developed by Basak Tamer | Data Science Portfolio")