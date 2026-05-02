import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

st.set_page_config(page_title="Podcast Predictor", page_icon="🎙️")

def load_assets():
    model = joblib.load('xgb_podcast_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

def main():
    st.title("🎙️ Podcast Listening Time Predictor")
    
    col1, col2 = st.columns(2)
    with col1:
        # Use the exact categories from your Kaggle training
        genre = st.selectbox("Genre", ["Society", "Comedy", "Technology", "Health", "History", "Business"])
        episode_len = st.number_input("Episode Length (Minutes)", min_value=1.0, value=45.0)
        sentiment_label = st.selectbox("Episode Sentiment", ["Positive", "Neutral", "Negative"])
        ads = st.number_input("Number of Ads", min_value=0, max_value=20, value=2)

    with col2:
        host_pop = st.slider("Host Popularity (%)", 0, 100, 75)
        guest_pop = st.slider("Guest Popularity (%)", 0, 100, 50)
        pub_day = st.selectbox("Publication Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        pub_time = st.selectbox("Publication Time", ["Morning", "Afternoon", "Evening", "Night"])

    if st.button("Predict"):
        try:
            model, scaler, feature_names = load_assets()

            # 1. Map inputs to match your feature engineering
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
            sentiment_map = {"Positive": 0.8, "Neutral": 0.5, "Negative": 0.2}
            
            # 2. Create a template of ZEROS based on your training columns
            # This is the most important step to stop getting '2.71'
            df_final = pd.DataFrame(0, index=[0], columns=feature_names)
            
            # 3. Inject numerical values into the correct columns
            # Verify these names match your Kaggle 'X.columns' exactly
            if 'Episode_Length_minutes' in df_final.columns:
                df_final.at[0, 'Episode_Length_minutes'] = episode_len
            if 'Episode_Sentiment_Score' in df_final.columns:
                df_final.at[0, 'Episode_Sentiment_Score'] = sentiment_map[sentiment_label]
            if 'Number_of_Ads' in df_final.columns:
                df_final.at[0, 'Number_of_Ads'] = min(ads, 3) # Your clipping logic
            if 'Host_Popularity_percentage' in df_final.columns:
                df_final.at[0, 'Host_Popularity_percentage'] = host_pop
            if 'Guest_Popularity_percentage' in df_final.columns:
                df_final.at[0, 'Guest_Popularity_percentage'] = guest_pop
            if 'Publication_Day_Num' in df_final.columns:
                df_final.at[0, 'Publication_Day_Num'] = day_map[pub_day]
            if 'Publication_Time_Ordinal' in df_final.columns:
                df_final.at[0, 'Publication_Time_Ordinal'] = time_map[pub_time]
            if 'is_weekend' in df_final.columns:
                df_final.at[0, 'is_weekend'] = 1 if day_map[pub_day] >= 5 else 0

            # 4. Handle the One-Hot Encoded Genre[cite: 1]
            genre_col = f"Genre_{genre}"
            if genre_col in df_final.columns:
                df_final.at[0, genre_col] = 1

            # 5. Scale and Predict[cite: 1]
            X_scaled = scaler.transform(df_final)
            
            # If using XGBoost, it's safer to use DMatrix or direct predict
            raw_pred = model.predict(X_scaled)[0]
            
            # Logical constraint: can't listen more than the length[cite: 1]
            final_pred = max(0, min(raw_pred, episode_len))

            st.success(f"### Predicted Listening Time: {final_pred:.2f} minutes")
            st.write(f"**Completion Rate:** {(final_pred/episode_len):.1%}")
            st.progress(min(final_pred/episode_len, 1.0))

        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()