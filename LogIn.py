import streamlit as st
import streamlit_parameters
from streamlit_login_auth_ui.widgets import __login__
from st_pages import hide_pages
import time

__login__obj = __login__(auth_token = st.secrets["auth_token"], 
                    company_name = "Stock Prediction App Team",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()

if LOGGED_IN == True:
    st.session_state['Login'] = True
    st.markdown("Your Streamlit Application Begins here!")
    with st.spinner('Wait for it...'):
        time.sleep(5)
        st.switch_page("pages/01Hello.py")
else:
    st.session_state['Login'] = False