import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


stocks = pd.read_csv('List_of_Stocks.csv')
stocks = stocks.drop_duplicates()

def dele(str):
    if 'common stock' in str.lower():
        str = str.replace('Common Stock', '')
    return str
stocks['Name'] = stocks['Name'].apply(lambda x: dele(x))
st.title("Stock Trend Prediction")

user_input = 'AAPL'
user_input = st.selectbox("Select the movie you like", tuple(stocks['Name'].values))

user_input = stocks.loc[stocks['Name'] == user_input, 'Symbol'].values[0] 

col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start Date")
with col2:
    end = st.date_input("End Date")

data = yf.download(user_input, start, end)
data = data.reset_index()

st.subheader('Interactive Chart')
fig_bar = px.line(data, y='Close', x='Date')
st.plotly_chart(fig_bar)


fig_candlestick = go.Figure(data=[go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])

st.subheader('Candlestick Chart')
fig_candlestick.update_layout(xaxis_title='Date',
                              yaxis_title='Price')

st.plotly_chart(fig_candlestick)




# Describing Data
st.subheader('Data from 2010 - 2019')
st.dataframe(data.describe(), use_container_width=True)

# Visulaization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(ma100, 'r')
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# """
# Already trained the model, doesn't need this part

# # Splitting Data into x_train and y_train
# x_train = []
# y_train = []

# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i,0])
    
# x_train, y_train = np.array(x_train),np.array(y_train)
# """

# Load model
model = load_model('keras_model.h5')

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

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = ' Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.xlabel('Price')
plt.legend()
st.pyplot(fig2)