"""
Web-based Demo App for the Streamlit-NewsAPI Connector.

Author:
    @dcarpintero : https://github.com/dcarpintero
"""
import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf  
import datetime as dt
#Variables
max_end = dt.datetime.now().date()
min_start = dt.datetime(2012,1,1).date()

#Título
st.title('Stock Price Prediction App')

#Preparacion Datos

yf.pdr_override()
Ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')
stock_data = yf.Ticker(Ticker)

start_column, end_column = st.columns(2)

with start_column:
    start = st.sidebar.date_input("Start date", min_value=min_start, max_value=max_end, value=min_start)

with end_column:
    end = st.sidebar.date_input("End date", min_value=start, max_value=max_end, value=max_end)
    
#Fetching the data
df = pdr.get_data_yahoo(Ticker,start,end)

#Describing data
st.subheader(stock_data.info['longName']+"("+Ticker + ") Stock data from " + str(start))

for i in range(7):
    st.header(stock_data.news[i]['title'], anchor=stock_data.news[i]['link'])
    st.subheader(stock_data.news[i]['publisher'])

        