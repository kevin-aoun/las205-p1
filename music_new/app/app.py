import streamlit as st

from music_new.app.tabs import *
from music_new.app.sidebar import render_config_sidebar
from music_new.core import init_config, check_model_files

def run_app2():
    """Main application entry point"""
    # Initialize configuration in session state
    init_config()
    config = st.session_state.config

    # Initialize session state variables
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False
    if 'file_processed' not in st.session_state:
        st.session_state['file_processed'] = False
    if 'current_data' not in st.session_state:
        st.session_state['current_data'] = None
    if 'trained_model' not in st.session_state:
        st.session_state['trained_model'] = None
    if 'label_encoder' not in st.session_state:
        st.session_state['label_encoder'] = None
    if 'show_training_report' not in st.session_state:
        st.session_state['show_training_report'] = False

    # Set page title
    st.title(config['app']['title'])

    # Handle configuration sidebar
    config_updated = render_config_sidebar()

    # Reload app if configuration was updated
    if config_updated:
        st.rerun()

    # Check if a saved model exists
    latest_model = check_model_files(return_full_paths=False)
    model_exists = latest_model is not None and config['model']['use_saved_model']

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data & Model", "Prediction", "Training Report", "Explore Data"])

    with tab1:
        render_data_model_tab(model_exists, latest_model)

    with tab2:
        render_prediction_tab(model_exists)

    with tab3:
        render_report_tab(model_exists)

    with tab4:
        render_explore_tab()

if __name__ == "__main__":
    run_app2()