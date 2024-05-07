#python -m streamlit run app.py
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf  
import streamlit as st
import datetime as dt
import GoogleNews as GN

#Variables
max_end = dt.datetime.now().date()
min_start = dt.datetime(2012,1,1).date()

#Collecting data...
yf.pdr_override()
Ticker1 = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')
Ticker2 = st.sidebar.text_input('Enter the stock ticker:', 'TSLA')
stock_data1 = yf.Ticker(Ticker1)
stock_data2 = yf.Ticker(Ticker2)

start_column, end_column = st.columns(2)

with start_column:
    start = st.sidebar.date_input("Start date", min_value=min_start, max_value=max_end, value=min_start)

with end_column:
    end = st.sidebar.date_input("End date", min_value=start, max_value=max_end, value=max_end)
#Título
st.title('Stock Market News')
st.write(GN())