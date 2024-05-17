import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf  
import datetime as dt
import requests
import io
from PIL import Image 

# Variables
max_end = dt.datetime.now().date()
min_start = dt.datetime(2012, 1, 1).date()

# Function to show image from link
def show_image_from_link(link, caption=None):
    r = requests.get(link)
    img = Image.open(io.BytesIO(r.content))
    st.image(img, caption=caption, use_column_width=True)

# Title
st.title('Stock Price Prediction App')

# Prepare Data
yf.pdr_override()

# Function to fetch data
def fetch_data(ticker):
    stock_data = yf.Ticker(ticker)
    df = pdr.get_data_yahoo(ticker)
    return stock_data, df

# Sidebar for entering stock ticker
ticker_input = st.sidebar.text_input('Enter the stock ticker:', st.query_params.get("ticker", "AAPL"))

# Fetching the data
stock_data, df = fetch_data(ticker_input)

# Describe data
st.subheader(stock_data.info['longName'] + "(" + ticker_input + ")")


# Function to display news
def mostrar_noticia(noticia):
    with st.container():
        st.header(noticia['title'], anchor=noticia['link'])
        st.caption(noticia['publisher'] + " for " + noticia['publisher'])
        st.write(noticia['link'])
        
        # Check if 'thumbnail' exists and has proper resolutions
        if 'thumbnail' in noticia and noticia['thumbnail'] and 'resolutions' in noticia['thumbnail'] and noticia['thumbnail']['resolutions']:
            show_image_from_link(noticia['thumbnail']['resolutions'][0]['url'])
        
        # Show related tickers
        if noticia['relatedTickers']:
            tickers = ', '.join([f"[{ticker}](/News2?ticker={ticker})" for ticker in noticia['relatedTickers']])
            st.markdown(f"**Related Tickers:** {tickers}")

# Iterate over news and display
for noticia in stock_data.news:
    mostrar_noticia(noticia)


