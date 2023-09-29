import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from PIL import Image
import requests.exceptions
import joblib


# Load animation
url = 'https://media.tenor.com/bu0w-cRvyU8AAAAC/welcome.gif'
st.image(url,use_column_width=True)


st.markdown("<h1 style='white-space: nowrap; color: darkblue;'>  Stock  Trend  Prediction  System </h1>", unsafe_allow_html=True)

# Load image
image2 = Image.open('C:/Users/ADMIN/Downloads/WELCOME_PICS2.jpg')
# Display image
st.image(image2, use_column_width=True)



st.write("\n")

user_input = st.text_input("Please Enter Stock Ticker", "AAPL")

try:
    ticker = yf.Ticker(user_input)
    if not ticker.info:
        st.warning(f"No information found for ticker symbol {user_input}")
    else:

        start = '2013-05-06'
        end = '2023-05-01'

        df = yf.download(user_input, start=start, end=end)
        print(df.head())


        # Describing Data
        st.header("Data from 2013 - 2023")
        st.write(df.describe())


        # visualizations
        st.header("Closing Price vs Time chart")
        fig = plt.figure(figsize=(15, 8))
        plt.plot(df.Close)
        st.pyplot(fig)

        st.header("Closing Price vs Time chart with 300MA")
        ma300 = df.Close.rolling(300).mean()
        fig = plt.figure(figsize=(15, 8))
        plt.plot(ma300)
        plt.plot(df.Close)
        st.pyplot(fig)


        st.header("Closing Price vs Time chart with 300MA & 500MA")
        ma300 = df.Close.rolling(300).mean()
        ma500 = df.Close.rolling(500).mean()
        fig = plt.figure(figsize=(15, 8))
        plt.plot(ma300, "y")
        plt.plot(ma500, "r")
        plt.plot(df.Close)
        st.pyplot(fig)


        # Splitting Data Into Training & Testing
        data_training = pd.DataFrame(df["Close"][0 : int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.70) : int(len(df))])
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0, 1))

        data_training_array = scaler.fit_transform(data_training)


        # Load Model
        model = load_model("keras_model.h5")
        

        # Load the Random Forest model
        rf_model = joblib.load('random_forest_model.pkl')

        # Use the loaded model for predictions or other operations
        # Example:
        


        # Testing Part
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_

        scale_factor = 1 / scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Retrieve the scaling factor from the scaler
        scale_factor_rf =1/scaler[0]
        x_test_rf = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2])
        y_test_pred_rf = rf_model.predict(x_test_rf)
        # Scale the predictions from the Random Forest model
        y_test_pred_rf_scaled = y_test_pred_rf * scale_factor_rf


        # Final Graph
        st.markdown("<h2 style='white-space: nowrap; color: darkblue;'>Predicted vs Original</h3>", unsafe_allow_html=True)
        fig2 = plt.figure(figsize=(15, 8))
        plt.plot(y_test, "b", label="Original Price")
        plt.plot(y_predicted, "r", label="Predicted price by LSTM MODEL")
        plt.plot(y_test_pred_rf_scaled, 'g', label = 'Predicted price by RANDOM FOREST')
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()

        st.pyplot(fig2)
        st.write("\n")
        image3 = Image.open('C:/Users/ADMIN/Downloads/pic3.jpg')
        st.image(image3,use_column_width=True)

        pass

except requests.exceptions.HTTPError as e:
     # Display imageS & WARNINGS
    st.warning(f"please try again & enter a valid stock ticker")
   
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Error.svg/2198px-Error.svg.png"
    st.image(image_url, width=200)


    st.warning(f"Error fetching information for ticker symbol {user_input}: {str(e)}")
    image3 = Image.open('C:/Users/ADMIN/Downloads/pic3.jpg')

    st.image(image3, use_column_width=True)


   
