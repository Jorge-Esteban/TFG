import streamlit as st
import pages

def main():

    # Sidebar
    st.sidebar.title("Navigation")
    st.title("Welcome Stock prediction! 👋")
    st.subheader("Web-App created by Jorge Esteban Gil")
    st.sidebar.success("Select a demo above.")

if __name__ == "__main__":
    main()
