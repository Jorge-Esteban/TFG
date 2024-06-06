#python -m streamlit run app.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf  
import streamlit as st
import datetime as dt
import netron
from pandas_datareader import data as pdr
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset, time_step = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
                   a = dataset[i:(i+time_step),0]
                   dataX.append(a)
                   dataY.append(dataset[i + time_step,0])
    return np.array(dataX),np.array(dataY)


st.set_page_config(page_icon="🔮", page_title='StockAIPrediction')
#Variables
max_end = dt.datetime.now().date()
min_start = dt.datetime(2012,1,1).date()
    
#Título
st.title('Stock Price Prediction App')

#Preparacion Datos
yf.pdr_override()
Ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL').rstrip().strip()
#try:
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

#Visualizaciones
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

# Customize the historical data graph
fig.update_layout(xaxis_rangeslider_visible=False)

# Use the native streamlit theme.
st.plotly_chart(fig, use_container_width=True)

#Closing price vs 100MA
st.subheader('Closing price vs Time chart with 100MA')
df['MA100'] = df['Close'].rolling(100).mean()
fig = px.line(df, 
            x=df.index, 
            y=['Close', 'MA100'], 
            labels={'value': 'Price', 'variable': 'Legend'}, 
            title='Closing Price vs 100-Day Moving Average',
            color_discrete_map={'MA100': '#F15050', 'Close':'#50BBD8'})
st.plotly_chart(fig, use_container_width=True)


# Closing price vs 100MA & 200MA
st.subheader('Closing Price vs Time chart with 100MA & 200MA')
df['MA200'] = df['Close'].rolling(200).mean()
fig = px.line(df, 
            x=df.index, 
            y=['Close', 'MA100', 'MA200'], 
            labels={'value': 'Price', 'variable': 'Legend'}, 
            title='Closing Price vs 100-Day Moving Average vs 200-Day moving Average',
            color_discrete_map={'MA200': '#47F388','MA100': '#F15050', 'Close':'#50BBD8'})
st.plotly_chart(fig, use_container_width=True)


#######################################
####        THE PREDICTION         ####
#######################################
st.markdown("## **Stock Prediction**")
#Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#Splitting Data into x_train and y_train
x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)


#Load model
select_model = st.selectbox('Choose your model',('Slow and scatter','Fast and accurate') )
if select_model == 'Slow and scatter' : 
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


    #Final Graph
    y_test_df = pd.DataFrame(y_test)
    y_predicted_df = pd.DataFrame(y_predicted)
    # Create a plot for the stock prediction
    fig_pred = go.Figure(
        data=[
            go.Scatter(
                x=df.index,
                y=y_test_df[0],
                name="Train",
                mode="lines",
                line=dict(color="blue"),
            ),
            go.Scatter(
                x=df.index,
                y=y_predicted_df[0],
                name="Forecast",
                mode="lines",
                line=dict(color="red"),
            )
        ]
    )


    # Show the candlestick chart
    st.plotly_chart(fig_pred)
    
else :
    #Second model
    model = load_model('stock_prediction2.h5') 
    df2 = df.reset_index()['Close']
    scaler = MinMaxScaler()
    df2 = scaler.fit_transform(np.array(df2).reshape(-1,1))
    
    train_size = int(len(df2)*0.65)
    test_size = len(df2) - train_size
    train_data,test_data = df2[0:train_size,:],df2[train_size:len(df2),:1]
    
    time_step = 100
    X_train,Y_train =  create_dataset(train_data,time_step)
    X_test,Y_test =  create_dataset(test_data,time_step)
    
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    
    look_back = 100
    
    trainPredictPlot = np.empty_like(df2)
    trainPredictPlot[:,:] = np.nan
    trainPredictPlot[look_back : len(train_predict)+look_back,:] = train_predict
    
    testPredictPlot = np.empty_like(df2)
    testPredictPlot[:,:] = np.nan
    testPredictPlot[len(train_predict)+(look_back)*2 + 1 : len(df2) - 1,:] = test_predict
    
    x_values = np.arange(len(df2))

    # Plotly trace for df2
    trace_df2 = go.Scatter(
        x=x_values,
        y=scaler.inverse_transform(df2).flatten(),
        mode='lines',
        name=Ticker,
        line=dict(color='blue')
    )

    # Plotly trace for trainPredictPlot
    trace_train = go.Scatter(
        x=x_values,
        y=trainPredictPlot.flatten(),
        mode='lines',
        name='Train Data',
        line=dict(color='green')
    )

    # Plotly trace for testPredictPlot
    trace_test = go.Scatter(
        x=x_values,
        y=testPredictPlot.flatten(),
        mode='lines',
        name='Test data',
        line=dict(color='red') 
    )

    # Create the plotly figure
    fig2 = go.Figure([trace_df2, trace_train, trace_test])

    # Show the plot
    st.plotly_chart(fig2)
    
    
if st.button(label="Go to model figure and stats 🌐"):
    if select_model == 'Slow and scatter' : 
        netron.start('stock_prediction.h5') 
    else :
        netron.start('stock_prediction2.h5') 
