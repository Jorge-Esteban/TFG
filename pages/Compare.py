#python -m streamlit run app.py
import pandas as pd
from pandas_datareader import data as pdr
import LogIn
import yfinance as yf  
from tensorflow.python.keras.models import load_model
import streamlit as st
import datetime as dt
import plotly.express as px

st.set_page_config(page_icon="🔀", page_title='StockComparison')
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

try: 
    
    #Título
    st.title('Stock Price Prediction App')

    ##Collecting data...
    yf.pdr_override()
    Ticker1 = st.sidebar.text_input('Enter the stock ticker:', 'AAPL').rstrip().strip()
    Ticker2 = st.sidebar.text_input('Enter the stock ticker:', 'TSLA').rstrip().strip()
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
    Recommendation_df = pd.DataFrame(columns=['Metric', 'Stock1', 'Stock2'])
    Key_statistics_df = pd.DataFrame(columns=['Metric', 'Stock1', 'Stock2'])
    df = pd.DataFrame()


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
        '20-Day Moving Average' : lambda x: x['Close'].rolling(20).mean().iloc[-1],
        '50-Day Moving Average' : lambda x: x['Close'].rolling(50).mean().iloc[-1],
        '100-Day Moving Average' : lambda x: x['Close'].rolling(100).mean().iloc[-1],
        'Percent change':  lambda x : x['Close'].pct_change().iloc[-1]
    }
    metrics_statistics= {
        'Market Cap': lambda x: x.info['marketCap'],
        'Shares Outstanding' : lambda x: x.info['sharesOutstanding'],
        'Last Anual net Income': lambda x: x.financials.iloc[23,0],
        'Last Quarter Performance': lambda x: x.info['mostRecentQuarter']
    }
    metric_recommendation= {
        'Strong buy': lambda x: x.recommendations['strongBuy'].iloc[0],
        'Buy': lambda x: x.recommendations['buy'].iloc[0], 
        'Hold': lambda x: x.recommendations['hold'].iloc[0],
        'Sell': lambda x: x.recommendations['sell'].iloc[0],
        'Strong sell': lambda x: x.recommendations['strongSell'].iloc[0]
    }
    for metric_name, metric_func in metrics_compare.items():
        Compare_df.loc[len(Compare_df)] = [metric_name, metric_func(stock_data1), metric_func(stock_data2)]

    for metric_name, metric_func in metrics_technical.items():
        Technical_df.loc[len(Technical_df)] = [metric_name, metric_func(df1), metric_func(df2)]
                                                                                                        
    for metric_name, metric_func in metrics_statistics.items():
        Key_statistics_df.loc[len(Key_statistics_df)] = [metric_name, metric_func(stock_data1), metric_func(stock_data2)] 
        
    for metric_name, metric_func in metric_recommendation.items():
        Recommendation_df.loc[len(Recommendation_df)] = [metric_name, metric_func(stock_data1), metric_func(stock_data2)]
        
    Compare_df.set_index('Metric', inplace=True)
    Technical_df.set_index('Metric', inplace=True)
    Key_statistics_df.set_index('Metric', inplace=True)
    Recommendation_df.set_index('Metric', inplace=True)

    Compare_df.loc['Volume'].apply(format_shares_money)
    Compare_df.loc['10-Day Average Volume'].apply(format_shares_money)
    Compare_df.rename(columns={'Stock1': stock_data1.info['longName'], 'Stock2': stock_data2.info['longName']}, inplace=True)


    Technical_df.rename(columns={'Stock1': stock_data1.info['longName'], 'Stock2': stock_data2.info['longName']}, inplace=True)
    Technical_df.loc['Percent change'] = Technical_df.loc['Percent change'].round(3)
    Technical_df.loc['Percent change'] = Technical_df.loc['Percent change'].astype(str) + '%'


    Recommendation_df.rename(columns={'Stock1': stock_data1.info['longName'], 'Stock2': stock_data2.info['longName']}, inplace=True)

    Key_statistics_df.loc['Market Cap'].apply(format_shares_money)
    Key_statistics_df.loc['Shares Outstanding'].apply(format_shares_money)
    Key_statistics_df.loc['Last Anual net Income'].apply(format_shares_money)
    Key_statistics_df.rename(columns={'Stock1': stock_data1.info['longName'], 'Stock2': stock_data2.info['longName']}, inplace=True)


    with st.container(border=True):

        if 'open' and 'dayLow' and 'dayHigh' and 'previousClose' and 'volume' and 'averageDailyVolume10Day' and 'industry' and 'sector' in stock_data1.info and stock_data2 :
            st.subheader('Basic Comparison', divider=True)
            st.table(Compare_df)
            st.divider()
            
            st.subheader('Technical Comparison', divider=True)
            st.table(Technical_df)
            st.divider()
        if 'marketCap' and 'sharesOutstanding' and 'mostRecentQuarter' in stock_data1.info :
            st.subheader('Key Statistics', divider=True)
            st.table(Key_statistics_df)
            st.divider()
        if 'strongBuy' and 'buy' and 'hold' and 'sell' and 'strongSell' in stock_data1.recommendations :
            st.subheader('Expert reccomendations', divider=True)
            st.table(Recommendation_df)


    #Closing price vs 100MA
    st.subheader(stock_data1.info['longName'] + 'vs ' + stock_data2.info['longName'])
    show_df = pd.DataFrame({
        'Stock1': df1['Close'].rolling(100).mean(),
        'Stock2': df2['Close'].rolling(100).mean()
    })
    fig = px.line(show_df, 
                x=show_df.index, 
                y=['Stock1', 'Stock2'], 
                labels={'value': 'Price', 'variable': 'Legend'}, 
                title=stock_data1.info['longName'] + ' vs ' + stock_data2.info['longName'] + ' 100-day Rolling Averages',
                color_discrete_map={'Stock1': '#F15050', 'Stock2': '#50BBD8'})

    st.plotly_chart(fig, use_container_width=True)
    
except : 
   st.write("Sorry, either one or both of the selected stock doesn't exist or there is no data. Try again please.")