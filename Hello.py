import streamlit as st
import pages

def main():

    # Sidebar
    st.sidebar.title("Navigation")
    st.write("# Welcome Stock prediction! 👋")
    st.sidebar.success("Select a demo above.")

if __name__ == "__main__":
    main()
