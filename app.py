import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("Select a page", ("Home", "Predict", ))

if page == "Home":
    show_explore_page()
else:
    show_predict_page()

