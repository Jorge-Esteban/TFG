#python -m streamlit run app.py
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf  
from keras.models import load_model
import streamlit as st
import datetime as dt

#Variables
max_end = dt.datetime.now().date()
min_start = dt.datetime(2012,1,1).date()

#Título
st.title('Stock Price Prediction App')

##Collecting data...
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
    
#Fetching the data
df1 = pdr.get_data_yahoo(Ticker1,start,end)
df2 = pdr.get_data_yahoo(Ticker2,start,end)

#Describing data
st.subheader(stock_data1.info['longName']+"("+Ticker1 + ") Stock data from " + str(start))
st.table(df1.describe())


#Volatility
volatility = round(df1['Close'].pct_change().std() * np.sqrt(252), 2) # Annualized volatility assuming 252 trading days per year
def VolatilityColor(number):
    if number > 0.75:  # Example threshold for high volatility
        return 'red'
    elif number < 0.25:  # Example threshold for low volatility
        return 'normal'
color = VolatilityColor(volatility)

#Market Capitalization
last_close_price = df1['Close'][-1]
shares_outstanding = 10_000_000  # Example number of shares outstanding
market_cap = format(last_close_price * shares_outstanding, '.2f')
    
#Useful measures
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Volatility')
    st.write(volatility)

with col2:
    st.subheader('Market Capitalization')
    st.write(market_cap)

with col3:
    # Return avergae Daily Percentage Change
    st.subheader('Avg. Daily Percentage Change')
    st.write(round(df1['Close'].pct_change().mean(),5))
##
###
####
#Second stock
st.subheader(stock_data2.info['longName']+"("+Ticker2 + ") Stock data from " + str(start))
st.table(df2.describe())   

    #Volatility
volatility2 = round(df2['Close'].pct_change().std() * np.sqrt(252), 2) # Annualized volatility assuming 252 trading days per year
color = VolatilityColor(volatility2)

#Market Capitalization
last_close_price = df2['Close'][-1]
shares_outstanding = 10_000_000  # Example number of shares outstanding
market_cap = format(last_close_price * shares_outstanding, '.2f')
    
#Useful measures
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Volatility')
    st.write(volatility2)

with col2:
    st.subheader('Market Capitalization')
    st.write(market_cap)

with col3:
    # Return avergae Daily Percentage Change
    st.subheader('Avg. Daily Percentage Change')
    st.write(round(df2['Close'].pct_change().mean(),5))
    
#Visualizaciones

    #Stock comparison
st.subheader(stock_data1.info['longName'] + " vs " + stock_data2.info['longName'])
fig = plt.figure(figsize=(10,6))
plt.plot(df1.Close)
plt.plot(df2.Close, 'g',label=Ticker2+'Closing price')
plt.plot(df1.Close,'b', label=Ticker1+'Closing price')
plt.legend()
plt.xlabel("Time", fontsize = 20)
plt.ylabel("Price", fontsize = 20)
st.pyplot(fig)


