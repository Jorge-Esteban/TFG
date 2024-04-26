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

#Visualizaciones
st.subheader('Closing price vs Time chart')
ma100 = df.Close.rolling(100).mean
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)
