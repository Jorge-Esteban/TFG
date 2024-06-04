import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
from st_pages import hide_pages
import time
__login__obj = __login__(auth_token = "pk_prod_0EDPKCYQBF472VHQ6J6FCNE24K4Q", 
                    company_name = "Stock Prediction App Team",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()

if LOGGED_IN == True:
    st.markdown("Your Streamlit Application Begins here!")
    with st.spinner('Wait for it...'):
        time.sleep(5)
        st.switch_page("pages/Hello.py")