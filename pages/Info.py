import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
from tensorflow.python.keras.models import load_model
import streamlit as st
import datetime as dt
import streamlit_card as stc
from streamlit_extras.grid import grid
import requests
import streamlit.components.v1 as components
#Variables
end = dt.datetime.now().date()
start = dt.datetime(2012,1,1).date()

#Título
st.title('Stock Price Prediction App')

#Preparacion Datos

yf.pdr_override()
Ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')
stock_data = yf.Ticker(Ticker)
df_instHolders = stock_data.institutional_holders
df_balancesheet = stock_data.balancesheet

class Tweet(object):
    def __init__(self, s, embed_str=False):
        if not embed_str:
            # Use Twitter's oEmbed API
            # https://dev.twitter.com/web/embedded-tweets
            api = "https://publish.twitter.com/oembed?url={}".format(s)
            response = requests.get(api)
            self.text = response.json()["html"]
        else:
            self.text = s

    def _repr_html_(self):
        return self.text

    def component(self):
        return components.html(self.text, height=600)


t = Tweet("https://twitter.com/OReillyMedia/status/901048172738482176").component()

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
    
# new_column_names = [str(col)[:4] for col in df_balancesheet.columns]
# new_column_names
# df_balancesheet.rename(columns=new_column_names, inplace=True)
df_balancesheet = df_balancesheet.iloc[:,0]
st.table(df_balancesheet)

#Describing data
if 'longName' and 'website' in stock_data.info:
    st.subheader(stock_data.info['longName']+"("+Ticker + ") Stock data from " + str(start))
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
clean_institutional_holders(df_instHolders)
st.subheader('Major institutonial investors:')
st.table(
   df_instHolders.style.highlight_max(['Shares'], color='green')
)


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

