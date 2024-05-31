import streamlit as st
import requests

import requests

def trendingPosts():
    
    url = "https://finance-social-sentiment-for-twitter-and-stocktwits.p.rapidapi.com/get-social-trending/posts"

    querystring = {"social":"twitter","isCrypto":"false","timestamp":"24h","limit":"10"}

    headers = {
        "Content-Type": "application/json",
        "X-RapidAPI-Key": "ebc2ec0c66msh8dda3e993fd9656p15b077jsn24ab995c70df",
        "X-RapidAPI-Host": "finance-social-sentiment-for-twitter-and-stocktwits.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    st.write(response.json())

def MostChangedTickers():

    url = "https://finance-social-sentiment-for-twitter-and-stocktwits.p.rapidapi.com/get-social-change/posts"

    querystring = {"social":"twitter","isCrypto":"false","timestamp":"24h","limit":"10"}

    headers = {
        "Content-Type": "application/json",
        "X-RapidAPI-Key": "ebc2ec0c66msh8dda3e993fd9656p15b077jsn24ab995c70df",
        "X-RapidAPI-Host": "finance-social-sentiment-for-twitter-and-stocktwits.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    st.write(response.json())
    
def LiveFeed():

    url = "https://finance-social-sentiment-for-twitter-and-stocktwits.p.rapidapi.com/get-social-feed"

    querystring = {"social":"twitter","tickers":"PLTR,BTC-USD","timestamp":"24h","limit":"10"}

    headers = {
        "Content-Type": "application/json",
        "X-RapidAPI-Key": "ebc2ec0c66msh8dda3e993fd9656p15b077jsn24ab995c70df",
        "X-RapidAPI-Host": "finance-social-sentiment-for-twitter-and-stocktwits.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    st.write(response.json())
    
LiveFeed()