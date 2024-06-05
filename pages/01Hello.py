import streamlit as st

def main():

    # Sidebar
    st.sidebar.title("Navigation")
    st.title("Welcome Stock prediction! 👋")
    st.subheader("Web-App created by Jorge Esteban Gil")
    st.sidebar.success("Select a demo above.")
    
    
    st.page_link("pages/02Info.py", label="Go to the Info page", icon="📈")
    st.page_link("pages/03Prediction.py", label="Go to the Prediction page", icon="🔮")
    st.page_link("pages/04Compare.py", label="Go to the Compare page", icon="🔀")
    st.page_link("pages/05News.py", label="Go to the News page", icon="📰")
    st.page_link("pages/06Twitter.py", label="Go to Twitter News page", icon="🐤")

if __name__ == "__main__":
    main()
