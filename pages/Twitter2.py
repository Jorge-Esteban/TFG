import requests
import streamlit as st
url = "https://twitter-api45.p.rapidapi.com/search.php"

querystring = {"query":"AAPL"}

headers = {
	"X-RapidAPI-Key": "ebc2ec0c66msh8dda3e993fd9656p15b077jsn24ab995c70df",
	"X-RapidAPI-Host": "twitter-api45.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

st.write(response)