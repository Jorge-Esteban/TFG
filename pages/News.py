import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
import requests
from PIL import Image
import io
import yfinance as yf 

# Función para mostrar imágenes desde enlaces
def show_image_from_link(link, caption=None):
    r = requests.get(link)
    img = Image.open(io.BytesIO(r.content))
    st.image(img, caption=caption, use_column_width=True)

# Función para mostrar las noticias relacionadas con un ticker
def mostrar_noticias(ticker):
    # Aquí podrías llamar a tu función o API para obtener las noticias relacionadas con el ticker
    pass

# Título
st.title('Stock Price Prediction App')

# Barra lateral para introducir el ticker
ticker_input = st.sidebar.text_input("Introduce el ticker", value="AAPL")
search_button = st.sidebar.button("Buscar")

# Si se hace clic en el botón de búsqueda, mostrar las noticias relacionadas con el ticker
if search_button:
    st.write(f"Noticias relacionadas con el ticker: {ticker_input}")
    mostrar_noticias(ticker_input)

# Preparación de datos
yf.pdr_override()
Ticker = ticker_input
stock_data = yf.Ticker(Ticker)

# Mostrar datos
df = pdr.get_data_yahoo(Ticker)

# Descripción de datos
st.subheader(stock_data.info['longName']+"("+Ticker)
st.write(stock_data.news)
