import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.python.keras.models import load_model
import streamlit as st
import datetime as dt
import streamlit_card as stc
from streamlit_extras.grid import grid
import requests
import streamlit.components.v1 as components
import LogIn
st.set_page_config(page_icon="📈", page_title='StockInfo')

#Functions
def format_shares_money(x):
        if x >= 1e9:  # If value is greater than or equal to 1 billion
            return '{:.2f}B'.format(x / 1e9)
        elif x >= 1e6:  # If value is greater than or equal to 1 million
            return '{:.2f}M'.format(x / 1e6)
        else:
            return '{:.2f}'.format(x)
        
def clean_institutional_holders(df_instHolders):
    
    df_instHolders.drop(columns=['Date Reported'], inplace=True)
    df_instHolders.set_index('Holder', inplace=True) 
    df_instHolders.rename(columns={'pctHeld':'% Held'}, inplace=True)
    df_instHolders['% Held'] = df_instHolders['% Held'] * 100
    df_instHolders['% Held'] = df_instHolders['% Held'].map('{:.2f}%'.format)

    # Formatting 'Shares' column dynamically
    df_instHolders['Shares'] = df_instHolders['Shares'].apply(format_shares_money)

def show_officer(officer):
        with st.container(border=True):
            st.subheader(officer['name'], divider=True)
            st.markdown(officer['title'])
            if 'age' in officer:
                st.write("Age:, ",officer['age'])
            if'totalPay' in officer:
                st.write("Anual pay: ", officer['totalPay'])
                
#Variables
end = dt.datetime.now().date()
start = dt.datetime(2012,1,1).date()

try:
    if LogIn.LOGGED_IN :
        #Título
        st.title('Stock Price Prediction App')

        #Preparacion Datos

        yf.pdr_override()
        Ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL').rstrip().strip()
        stock_data = yf.Ticker(Ticker)
        df = pdr.get_data_yahoo(Ticker,start,end)
        df_balancesheet = stock_data.balancesheet


        #Describing data
        if 'longName' and 'website' in stock_data.info:
            st.subheader(stock_data.info['longName']+"("+Ticker + ") Info:")
        elif 'longName' in stock_data:
            st.subheader(stock_data.info['longName']+"("+Ticker + ") Stock data from "+ str(start))
        if 'city' and 'address1' and 'country' and 'industry' and 'sector' in stock_data.info:
            st.write('Located at: ' + stock_data.info['address1'] , ", " , stock_data.info['city'] , ', ' , stock_data.info['country'])
            st.write('Industry: ' , stock_data.info['industry'])
            st.write('Sector: ', stock_data.info['sector'])
        if 'website' in stock_data.info:
            st.write("[Link to website](%s)" % stock_data.info['website'])
        if 'irWebsite' in stock_data.info:
            st.write("[Link to investor relations website](%s)" % stock_data.info['irWebsite'])
            
        #Description  
        if 'longBusinessSummary' in stock_data.info:            
            with st.container(border=True):
                st.write(stock_data.info['longBusinessSummary'])  

        #Holders
        try:
            df_instHolders = stock_data.institutional_holders   
            clean_institutional_holders(df_instHolders)
            st.subheader('Major institutonial investors:')
            st.table(
            df_instHolders
            )
        except: 
            st.write('**No institutional holders declared**')


        #Main officers
        st.subheader('Main officers:')
        if 'companyOfficers' in stock_data.info:           
            officers = stock_data.info['companyOfficers']
            num_columns = 2
            num_officers = len(officers)

            for i in range(0, num_officers, num_columns):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    if i + j < num_officers:
                        with cols[j]:
                            show_officer(officers[i + j])

        #Historical Data
        st.markdown("## **Historical Data**")

        # Create a plot for the historical data
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                )
            ]
        )
        #Balancesheet
        df_balancesheet = df_balancesheet.iloc[:,0]
        st.subheader('Balance sheet', divider=True)
        st.table(df_balancesheet)
    else:
        st.page_link("LogIn.py",label='Log In to access the app ')
except:
    st.write("Sorry, the selected stock doesn't exist or there is no data. Try again please.")
