import streamlit as st
import h5py 

import h5py
f = h5py.File('stock_prediction.h5','r')
for item in f.keys():
    st.write(item )#+ ":", f[item]