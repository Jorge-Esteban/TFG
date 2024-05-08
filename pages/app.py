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
from newspaper import Article

#Variables
max_end = dt.datetime.now().date()
min_start = dt.datetime(2012,1,1).date()

#Título
st.title('Stock Price Prediction App')

#Preparacion Datos

yf.pdr_override()
Ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')
stock_data = yf.Ticker(Ticker)

start_column, end_column = st.columns(2)

with start_column:
    start = st.sidebar.date_input("Start date", min_value=min_start, max_value=max_end, value=min_start)

with end_column:
    end = st.sidebar.date_input("End date", min_value=start, max_value=max_end, value=max_end)
    
#Fetching the data
df = pdr.get_data_yahoo(Ticker,start,end)

#Describing data
st.subheader(stock_data.info['longName']+"("+Ticker + ") Stock data from " + str(start))
st.table(df.describe())

#Volatility
volatility = round(df['Close'].pct_change().std() * np.sqrt(252), 2) # Annualized volatility assuming 252 trading days per year
color = 'off'
if volatility > 0.75:  # Example threshold for high volatility
    color = 'inverse'
elif volatility < 0.25:  # Example threshold for low volatility
    color = 'normal'

#Market Capitalization
last_close_price = df['Close'][-1]
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
    st.write(round(df['Close'].pct_change().mean(),5))
    
    
#Visualizaciones

    #Closing price vs 100MA
st.subheader('Closing price vs Time chart with 100MA')
fig = plt.figure(figsize=(10,6))
ma100 = df.Close.rolling(100).mean()
plt.plot(ma100)
plt.plot(df.Close)
plt.plot(ma100, 'r',label='100-Day Moving Average')
plt.plot(df.Close,'b', label='Closing Price')
plt.legend()
plt.xlabel("Time", fontsize = 20)
plt.ylabel("Price", fontsize = 20)
st.pyplot(fig)

    #Candle
fig = go.Figure(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))

    #Use the native streamlit theme.
st.plotly_chart(fig, use_container_width=True)

    #Closing price vs 100MA & 200MA
st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(10,6))
plt.plot(ma100, 'r', label = '100-Day Moving Average')
plt.plot(ma200, 'g', label = '200-Day Moving Average')
plt.plot(df.Close, 'b', label = 'Closing Price')
st.pyplot(fig)


#Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Splitting Data into x_train and y_train
x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)


#Load model
model = load_model('stock_prediction.h5')


#Testing part
past_100_days = data_training.tail(100)
final_df = past_100_days._append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor
st.write()
#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(10,6))
plt.plot(y_test, 'b', label= 'Original Price')
plt.plot(y_predicted, 'r', label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
Graph_img = plt.savefig('Predictions vs Original.png')

st.download_button("Download Graph", Graph_img)

increasing_color1 = '#00FF00'  # Green
decreasing_color1 = '#FF0000'  # Red
increasing_color2 = '#2A57A3'  # Blue
decreasing_color2 = '#E0A100'  # Gold
trace_original = go.Candlestick(x=df.index,
                                 open=y_test,
                                 high=y_test,
                                 low=y_test,
                                 close=y_test,
                                 increasing_line_color=increasing_color1,
                                 decreasing_line_color=decreasing_color1,
                                 name='Original Price')
trace_predicted = go.Candlestick(x=df.index,
                                 open=y_predicted,
                                 high=y_predicted,
                                 low=y_predicted,
                                 close=y_predicted,
                                 increasing_line_color=increasing_color2,
                                 decreasing_line_color=decreasing_color2,
                                 name='Predicted Price')

# Create layout
layout = go.Layout(title='Predictions vs Original',
                   xaxis=dict(title='Time'),
                   yaxis=dict(title='Price'))

# Create figure
fig_candlestick = go.Figure(data=[trace_original, trace_predicted], layout=layout)

# Show the candlestick chart
st.plotly_chart(fig_candlestick)