import streamlit as st
from weight.core import update_config

def render_config_sidebar():
    """
    Display and handle configuration sidebar.

    Returns:
        bool: True if configuration was updated, False otherwise
    """
    st.sidebar.header("Configuration")

    current_config = st.session_state.config

    updated_config = {
        'model': {**current_config['model']},
        'data': {**current_config['data']},
        'app': {**current_config['app']},
        'logging': {**current_config['logging']}
    }

    st.sidebar.subheader("Model Settings")
    updated_config['model']['save_model'] = st.sidebar.checkbox(
        "Save trained models",
        value=current_config['model']['save_model']
    )
    updated_config['model']['use_saved_model'] = st.sidebar.checkbox(
        "Use saved models",
        value=current_config['model']['use_saved_model']
    )
    updated_config['model']['n_estimators'] = st.sidebar.slider(
        "Number of estimators",
        10, 500,
        value=current_config['model']['n_estimators']
    )

    # Data settings section
    st.sidebar.subheader("Data Settings")
    updated_config['data']['save_uploads'] = st.sidebar.checkbox(
        "Save uploaded files",
        value=current_config['data']['save_uploads']
    )

    # App settings section
    st.sidebar.subheader("App Settings")
    updated_config['app']['default_height'] = st.sidebar.number_input(
        "Default height",
        min_value=0,
        max_value=100,
        value = current_config['app'].get('default_height', 100)
    )

    # Logging settings section
    st.sidebar.subheader("Logging Settings")
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    updated_config['logging']['level'] = st.sidebar.selectbox(
        "Logging level",
        log_levels,
        index=log_levels.index(current_config['logging']['level'])
    )

    # Save button
    if st.sidebar.button("Save Configuration"):
        # Check if anything actually changed
        changes_made = False
        for section in current_config:
            for key in updated_config[section]:
                if updated_config[section][key] != current_config[section][key]:
                    changes_made = True
                    break
            if changes_made:
                break

        if changes_made:
            if update_config(updated_config):
                st.sidebar.success("Configuration saved successfully!")
                return True
            else:
                st.sidebar.error("Failed to save configuration.")
        else:
            st.sidebar.info("No changes detected in configuration.")

    return False