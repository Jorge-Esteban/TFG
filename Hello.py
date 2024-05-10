import streamlit as st
import pages

def main():

    # Sidebar
    st.sidebar.title("Navigation")
    st.title("Welcome Stock prediction! 👋")
    st.subheader("Web-App created by Jorge Esteban Gil")
    st.sidebar.success("Select a demo above.")
    
    st.page_link("pages/app.py", label="Go to the Info page", icon="📈")
    st.page_link("pages/Compare.py", label="Go to the Compare page", icon="🔀")
    st.page_link("pages/News2.py", label="Go to the News page", icon="📰")
    
    

if __name__ == "__main__":
    main()
