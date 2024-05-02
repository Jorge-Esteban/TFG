import streamlit as st
import pages

def main():
    st.title("Streamlit Homepage")

    # Sidebar
    st.sidebar.title("Navigation")
    st.write("# Welcome to Streamlit! 👋")
    st.sidebar.success("Select a demo above.")

if __name__ == "__main__":
    main()
