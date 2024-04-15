import pandas as pd
import pandas.datareader as data
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf 
import datetime  
from keras.models import load_model
import streamlit as st

start = '2012-01-01'
end = datetime.datetime.now().strftime('%Y-%m-%d')

st.title('Stock Price Prediction App')
user_input = st.text_input('Enter the stock ticker:', 'AAPL')

df = data.datareader(user_input, 'yahoo', start, end)

#Describing data
st.subheader('Data from 2012 - To today')
st.write(df.describe())
