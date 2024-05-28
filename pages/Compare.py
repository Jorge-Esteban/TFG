#python -m streamlit run app.py
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf  
from tensorflow.python.keras.models import load_model
import streamlit as st
import datetime as dt
import plotly.express as px

#Variables
max_end = dt.datetime.now().date()
min_start = dt.datetime(2012,1,1).date()

#Funciones
def format_shares_money(x):
        if x >= 1e9:  # If value is greater than or equal to 1 billion
            return '{:.2f}B'.format(x / 1e9)
        elif x >= 1e6:  # If value is greater than or equal to 1 million
            return '{:.2f}M'.format(x / 1e6)
        else:
            return '{:.2f}'.format(x)
        
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

Compare_df = pd.DataFrame(columns=['Metric', 'Stock1', 'Stock2'])
Technical_df = pd.DataFrame(columns=['Metric', 'Stock1', 'Stock2'])
Key_statistics_df = pd.DataFrame(columns=['Metric', 'Stock1', 'Stock2'])

metrics_compare = {
    'Open': lambda x: x.info['open'],
    'Low ': lambda x: x.info['dayLow'],
    'High': lambda x: x.info['dayHigh'],
    'Last Close': lambda x: x.info['previousClose'],
    'Volume': lambda x: x.info['volume'],
    '10-Day Average Volume': lambda x: x.info['averageDailyVolume10Day'],
    'Industry':  lambda x: x.info['industry'],
    'Sector':  lambda x: x.info['sector']   
}
metrics_technical= {
    #'20-Day Moving Average' : lambda x: x['Close'].rolling(20).mean(),
}
metrics_statistics= {
    'Market Cap': lambda x: x.info['marketCap'],
    'Shares Outstanding' : lambda x: x.info['sharesOutstanding'],
    'Last Anual net Income': lambda x: x.financials.iloc[23,0],
    'Last Quarter Performance': lambda x: x.info['mostRecentQuarter']
}

for metric_name, metric_func in metrics_compare.items():
    Compare_df.loc[len(Compare_df)] = [metric_name, metric_func(stock_data1), metric_func(stock_data2)]

for metric_name, metric_func in metrics_technical.items():
    Technical_df.loc[len(Technical_df)] = [metric_name, metric_func(df1), metric_func(df1)]

for metric_name, metric_func in metrics_statistics.items():
    Key_statistics_df.loc[len(Key_statistics_df)] = [metric_name, metric_func(stock_data1), metric_func(stock_data2)] 
    
Compare_df.set_index('Metric', inplace=True)
Technical_df.set_index('Metric', inplace=True)
Key_statistics_df.set_index('Metric', inplace=True)

Compare_df.loc['Volume'].apply(format_shares_money)
Compare_df.loc['10-Day Average Volume'].apply(format_shares_money)

Key_statistics_df.loc['Market Cap'].apply(format_shares_money)
Key_statistics_df.loc['Shares Outstanding'].apply(format_shares_money)
Key_statistics_df.loc['Last Anual net Income'].apply(format_shares_money)


#Describing data
st.subheader(stock_data1.info['longName']+"("+Ticker1 + ") Stock data from " + str(start))
st.table(df1.describe())

st.subheader('Basic Comparison')
st.table(Compare_df)
st.divider()
st.subheader('Key Statistics')
st.table(Key_statistics_df)
##
###
####
#Second stock
st.subheader(stock_data2.info['longName']+"("+Ticker2 + ") Stock data from " + str(start))
st.table(df2.describe())   

    
#Visualizaciones

#Closing price vs 100MA
st.subheader(stock_data1.info['longName'] + 'vs ' + stock_data2.info['longName'])
df = pd.DataFrame()
df['Stock1'] = df1['Close'].rolling(100).mean()
df['Stock2'] = df2['Close'].rolling(100).mean()
fig = px.line(df, 
              x=df.index, 
              y=['Stock1', 'Stock2'], 
              labels={'value': 'Price', 'variable': 'Legend'}, 
              title=stock_data1.info['longName']+stock_data2.info['longName'],
              color_discrete_map={'Stock1': '#F15050', 'Stock2':'#50BBD8'})
st.plotly_chart(fig, use_container_width=True)
