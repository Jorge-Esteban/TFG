import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf  
import datetime as dt
import requests
import io
from PIL import Image 

st.set_page_config(page_icon="📰", page_title='StockNews')
# Variables
max_end = dt.datetime.now().date()
min_start = dt.datetime(2012, 1, 1).date()

# Function to show image from link
def show_image_from_link(link, caption=None):
    r = requests.get(link)
    img = Image.open(io.BytesIO(r.content))
    st.image(img, caption=caption, use_column_width=True)



# Prepare Data
yf.pdr_override()

# Function to fetch data
try:
    def fetch_data(ticker):
        stock_data = yf.Ticker(ticker)
        df = pdr.get_data_yahoo(ticker)
        return stock_data, df

# Sidebar for entering stock ticker
    ticker_input = st.sidebar.text_input('Enter the stock ticker:', st.query_params.get("ticker", "AAPL")).rstrip().strip()

# Fetching the data
    stock_data, df = fetch_data(ticker_input)

# Title

    st.title(stock_data.info['longName'] + "(" + ticker_input + ") Latest News")


    # Function to display news
    def show_news(noticia):
        with st.container(border=True):
            st.header(noticia['title'], anchor=noticia['link'], divider=True)
            st.caption(noticia['publisher'] + " for " + noticia['publisher'])
            # Check if 'thumbnail' exists and has proper resolutions
            if 'thumbnail' in noticia and noticia['thumbnail'] and 'resolutions' in noticia['thumbnail'] and noticia['thumbnail']['resolutions']:
                show_image_from_link(noticia['thumbnail']['resolutions'][0]['url'])
            st.write("[Link to the news](%s)" % noticia['link'])
            # Show related tickers
            st.divider()
            if noticia['relatedTickers']:
                tickers = ', '.join([f"[{ticker}](/News2?ticker={ticker})" for ticker in noticia['relatedTickers']])
                st.markdown(f"**Related Stocks:** {tickers}")

    # Iterate over news and display
    for noticia in stock_data.news:
        show_news(noticia)
except : 
    st.write("Sorry, the selected stock doesn't exist or there is no data. Try again please.")