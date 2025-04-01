import streamlit as st
from music.logs import logger
from music.prediction import predict_using_percentage, predict_using_ml
from music.app.utils import display_prediction_result


def render_prediction_tab(model_exists):
    """Render the Prediction tab"""
    if st.session_state.get('current_data') is None:
        st.info("Please upload a CSV file to make predictions.")
        return

    st.subheader("Enter Your Information")

    # Get user inputs
    gender = st.radio("Select your gender:", options=["Male", "Female"], index=0)
    gender_value = 1 if gender == "Male" else 0

    config = st.session_state.config
    age_min, age_max = config['app']['age_range']
    default_age = config['app']['default_age']
    age = st.slider("Select your age:", age_min, age_max, default_age)

    # Show which model will be used for prediction
    # Check if we're using a newly trained model or a selected model
    using_newly_trained = st.session_state.get('using_newly_trained_model', False)
    selected_model = st.session_state.get('selected_model')

    if using_newly_trained:
        st.success("Using newly trained model for prediction")
    elif selected_model and model_exists:
        st.info(f"Using selected model: {selected_model} for prediction")

    # Make predictions when button is clicked
    if st.button("Predict Music Preference"):
        generate_predictions(age, gender_value, model_exists)


def generate_predictions(age, gender_value, model_exists):
    """Generate and display all predictions"""
    data = st.session_state['current_data']
    can_use_ml = st.session_state.get('model_trained', False) or model_exists

    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)

    # Percentage-based prediction
    with col1:
        st.info("Based on Percentage Model")
        try:
            percentage_result = predict_using_percentage(data, age, gender_value)
            display_prediction_result(percentage_result)
        except Exception as e:
            st.error(f"Error in percentage prediction: {str(e)}")
            logger.error(f"Error in percentage prediction: {str(e)}", exc_info=True)

    # ML-based prediction
    with col2:
        st.info("Based on Machine Learning Model")

        if not can_use_ml:
            st.write("No ML model available for prediction")
        else:
            try:
                # Determine which model to use based on state
                using_newly_trained = st.session_state.get('using_newly_trained_model', False)
                in_memory_model = (st.session_state.get('trained_model') is not None and
                                   st.session_state.get('label_encoder') is not None)

                if using_newly_trained and in_memory_model:
                    # Use the newly trained model from memory
                    logger.info("Using newly trained model from memory")
                    ml_result = predict_using_ml(
                        None, age, gender_value,
                        st.session_state['trained_model'],
                        st.session_state['label_encoder']
                    )
                else:
                    # Use the selected model from disk
                    selected_model = st.session_state.get('selected_model')
                    logger.info(f"Using selected model from disk: {selected_model}")
                    ml_result = predict_using_ml(data, age, gender_value, model_filename=selected_model)

                display_prediction_result(ml_result)
            except Exception as e:
                st.error(f"Error in ML prediction: {str(e)}")
                logger.error(f"Error in ML prediction: {str(e)}", exc_info=True)

    logger.info(f"Predictions displayed for age={age}, gender={gender_value}")