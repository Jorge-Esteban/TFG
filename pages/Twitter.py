from collections import defaultdict, namedtuple
from htbuilder import div, big, h2, styles
from htbuilder.units import rem
from math import floor
from textblob import TextBlob
import altair as alt
import datetime
import functools
import pandas as pd
import re
import streamlit as st
import time
import tweepy
import requests
import streamlit.components.v1 as components
st.set_page_config(page_icon="🐤", page_title="Twitter Sentiment Analyzer")

consumer_key="PHvbXVcMWNhKZwL3p5CkNa0Az"
consumer_secret="6G8EGd9urlm5BT15WzyORVHz8rQkDqqlRiKouIwkBwlZhYNEqQ"
access_token="714872719393886208-RpOxyxuc6M0BajNvCg4FMs9xw5yDB12"
access_token_secret="by9SF8FEpzzlwJ8oSFMqIwmcqTLcBHlRcYK0n0uSSGkrh"
BearerToken = 'AAAAAAAAAAAAAAAAAAAAACREuAEAAAAA%2BafTUAhckfYMx1M5ymwK2cexTLQ%3DObJatBRgtOjfKU8uxbhi8F3Uh4TadH3xPmLU41KyZfKCNOQweu'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


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



if api.verify_credentials :
    st.write('Credentials work!')

# Define the search query
search_query = '$AAPL'

# Collect tweets
tweets = tweepy.Cursor(api.search_tweets, q=search_query, lang='en').items(5)
# Print the collected tweets
for tweet in tweets:
    st.write(tweet.full_text)
    st.write('-' * 50)# Collect tweets
    
tweets = tweepy.Cursor(api.search_tweets, q=search_query, lang='en').items(5)
# Print the collected tweets
for tweet in tweets:
    st.write(tweet.full_text)
    st.write('-' * 50)
    
