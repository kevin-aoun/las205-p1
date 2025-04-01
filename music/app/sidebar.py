import streamlit as st
from music.core import update_config


def render_config_sidebar():
    """
    Display and handle configuration sidebar.

    Returns:
        bool: True if configuration was updated, False otherwise
    """
    st.sidebar.header("Configuration")

    current_config = st.session_state.config

    # Keep track of whether any changes were made during this session
    if 'config_changed' not in st.session_state:
        st.session_state['config_changed'] = False

    # Create a deep copy for updated config
    updated_config = {
        'model': {**current_config['model']},
        'data': {**current_config['data']},
        'app': {**current_config['app']},
        'logging': {**current_config['logging']}
    }

    # Model settings section with change tracking
    st.sidebar.subheader("Model Settings")
    save_model = st.sidebar.checkbox(
        "Save trained models",
        value=current_config['model']['save_model'],
        key="save_model_checkbox"
    )
    updated_config['model']['save_model'] = save_model
    if save_model != current_config['model']['save_model']:
        st.session_state['config_changed'] = True

    use_saved_model = st.sidebar.checkbox(
        "Use saved models",
        value=current_config['model']['use_saved_model'],
        key="use_saved_model_checkbox"
    )
    updated_config['model']['use_saved_model'] = use_saved_model
    if use_saved_model != current_config['model']['use_saved_model']:
        st.session_state['config_changed'] = True

    n_estimators = st.sidebar.slider(
        "Number of estimators",
        10, 500,
        value=current_config['model']['n_estimators'],
        key="n_estimators_slider"
    )
    updated_config['model']['n_estimators'] = n_estimators
    if n_estimators != current_config['model']['n_estimators']:
        st.session_state['config_changed'] = True

    # Data settings section
    st.sidebar.subheader("Data Settings")
    save_uploads = st.sidebar.checkbox(
        "Save uploaded files",
        value=current_config['data']['save_uploads'],
        key="save_uploads_checkbox"
    )
    updated_config['data']['save_uploads'] = save_uploads
    if save_uploads != current_config['data']['save_uploads']:
        st.session_state['config_changed'] = True

    # App settings section
    st.sidebar.subheader("App Settings")
    default_age = st.sidebar.number_input(
        "Default age",
        min_value=0,
        max_value=100,
        value=current_config['app']['default_age'],
        key="default_age_input"
    )
    updated_config['app']['default_age'] = default_age
    if default_age != current_config['app']['default_age']:
        st.session_state['config_changed'] = True

    # Logging settings section
    st.sidebar.subheader("Logging Settings")
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level = st.sidebar.selectbox(
        "Logging level",
        log_levels,
        index=log_levels.index(current_config['logging']['level']),
        key="log_level_selectbox"
    )
    updated_config['logging']['level'] = log_level
    if log_level != current_config['logging']['level']:
        st.session_state['config_changed'] = True

    # Save button with feedback
    save_button = st.sidebar.button("Save Configuration")

    if save_button:
        if st.session_state['config_changed']:
            if update_config(updated_config):
                st.sidebar.success("Configuration saved successfully!")
                # Reset the change flag
                st.session_state['config_changed'] = False
                return True
            else:
                st.sidebar.error("Failed to save configuration.")
        else:
            st.sidebar.info("No changes detected in configuration.")

    # Show indicator if there are unsaved changes
    if st.session_state['config_changed']:
        st.sidebar.warning("You have unsaved changes")

    return False