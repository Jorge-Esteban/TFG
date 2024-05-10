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

#Fetching the data
df = pdr.get_data_yahoo(Ticker)

#Describing data
st.subheader(stock_data.info['longName']+"("+Ticker)

st.write(stock_data.news)
for i in range(len(stock_data.news)):
    st.title(stock_data.news[i]['title'], anchor=stock_data.news[i]['link'])
    st.subheader(stock_data.news[i]['publisher'])
    for j in range(len(stock_data.news[i]['relatedTickers'])):
        st.caption(stock_data.news[i]['relatedTickers'][j])

        