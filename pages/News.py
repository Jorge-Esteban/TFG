#python -m streamlit run app.py
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf  
import streamlit as st
import datetime as dt
from GoogleNews import GoogleNews

#Título
st.title('Stock Market News')

#Variables
max_end = dt.datetime.now().date()
min_start = dt.datetime(2012,1,1).date()

#Collecting data...
yf.pdr_override()
Ticker1 = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')
stock_data1 = yf.Ticker(Ticker1)

start_column, end_column = st.columns(2)

with start_column:
    start_date = st.sidebar.date_input("Start date", min_value=min_start, max_value=max_end, value=min_start)

with end_column:
    end_date = st.sidebar.date_input("End date", min_value=start_date, max_value=max_end, value=max_end)

#GoogleNews data
googlenews = GoogleNews(lang='en', period=7 ,encode='utf-8')
googlenews.enableException(True)
googlenews.search(stock_data1.info['longName'])
result = googlenews.results(sort=True)

for i in range(10):
    st.header(result[i]['title'])
    st.subheader(result[i]['media'])
    st.write(result[i]['desc'])
    st.image(result[i]['image'])