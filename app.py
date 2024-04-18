#python -m streamlit run app.py
import pandas as pd
import pandas_datareader as data
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf  
from keras.models import load_model
import streamlit as st
import datetime as dt

start = dt.datetime(2012,1,1)
end = dt.datetime(2024,1,1)
#datetime.datetime.now().strftime('%Y-%m-%d')

st.title('Stock Price Prediction App')
user_input = st.text_input('Enter the stock ticker:', 'AAPL')

#df = data.DataReader(user_input, 'yahoo', start, end)
df = data.DataReader( user_input, data_source='yahoo', start=start, end=end)

#Describing data
st.subheader('Data from 2012 - To today')
st.write(df.describe())