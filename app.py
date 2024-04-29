#python -m streamlit run app.py
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf  
from keras.models import load_model
import streamlit as st
import datetime as dt

#Variables
max_end = dt.datetime.now().date()
min_start = dt.datetime(2012,1,1).date()

#Título
st.title('Stock Price Prediction App')

#Preparacion Datos
yf.pdr_override()
user_input = st.text_input('Enter the stock ticker:', 'AAPL')
start_column, end_column = st.columns(2)

with start_column:
    start = st.date_input("Start date", min_value=min_start, max_value=max_end, value=min_start)

with end_column:
    end = st.date_input("End date", min_value=start, max_value=max_end, value=max_end)
    
#Fetching the data
df = pdr.get_data_yahoo(user_input,start,end)

#Describing data
st.subheader(user_input + " Stock data from " + str(start))
st.table(df.describe())

#Volatility
volatility = round(df['Close'].pct_change().std() * np.sqrt(252), 2) # Annualized volatility assuming 252 trading days per year
color = 'off'
if volatility > 0.75:  # Example threshold for high volatility
    color = 'inverse'
elif volatility < 0.25:  # Example threshold for low volatility
    color = 'normal'

#Market Capitalization
last_close_price = df['Close'][-1]
shares_outstanding = 10_000_000  # Example number of shares outstanding
market_cap = format(last_close_price * shares_outstanding, '.2f')
    
#Useful measures
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Volatility')
    st.metric(" ", volatility)

with col2:
    st.subheader('Market Capitalization')
    st.metric(" ", market_cap)

with col3:
    # Return avergae Daily Percentage Change
    st.subheader('Average Daily Percentage Change')
    st.metric(" ", round(df['Close'].pct_change().mean(),5))
    
#Visualizaciones


#Closing price vs 100MA
st.subheader('Closing price vs Time chart with 100MA')
fig = plt.figure(figsize=(10,6))
ma100 = df.Close.rolling(100).mean()
plt.plot(ma100)
plt.plot(df.Close)
plt.plot(ma100, label='100-Day Moving Average')
plt.plot(df.Close, label='Closing Price')
plt.legend()
plt.xlabel("Time", fontsize = 20)
plt.ylabel("Price", fontsize = 20)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(10,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close, 'b')
st.pyplot(fig)
