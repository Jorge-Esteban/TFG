import re
import streamlit as st
import requests
import streamlit.components.v1 as components
import yfinance as yf
st.set_page_config(page_icon="🐤", page_title="Twitter Feed")

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
    
def remove_media_suffix(url):
    # Define a regular expression pattern to match the /photo/1 or /video/1 part at the end of the URL
    pattern = r'(/(?:photo|video)/\d+)$'
    # Use re.sub to replace the pattern with an empty string
    cleaned_url = re.sub(pattern, '', url)
    return cleaned_url

def show_tweets(json):
    for i in range(10):
        if ('media' in json['timeline'][i]['entities']) and ('expanded_url' in json['timeline'][i]['entities']['media'][0]):
            x_url = json['timeline'][i]['entities']['media'][0]['expanded_url']
            x_url = remove_media_suffix(x_url)
            t = Tweet(x_url).component()

       
url = "https://twitter-api45.p.rapidapi.com/search.php"
headers = {
	"X-RapidAPI-Key": "ebc2ec0c66msh8dda3e993fd9656p15b077jsn24ab995c70df",
	"X-RapidAPI-Host": "twitter-api45.p.rapidapi.com"
}


try:
    
    Ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')
    stock_data = yf.Ticker(Ticker)
    querystring = {"query":'$'+Ticker}
    st.title(stock_data.info['longName'] + "(" + Ticker + ") Latest News on X")
    response = requests.get(url, headers=headers, params=querystring)
    response_json = response.json()
    show_tweets(response_json)

except:
    st.write("Sorry, the selected stock doesn't exist or there is no data. Try again please.")