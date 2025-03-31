import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from music.core import logger
from music.prediction import predict_using_percentage, predict_using_ml
from music.app.utils import display_percentage_results


def render_prediction_tab(model_exists):
    """
    Render the Prediction tab

    Args:
        model_exists (bool): Whether a usable model exists
    """
    # Only show prediction UI if a file has been uploaded
    if st.session_state.get('current_data') is not None:
        st.subheader("Enter Your Information")

        gender = st.radio("Select your gender:",
                          options=["Male", "Female"],
                          index=0)
        gender_value = 1 if gender == "Male" else 0

        config = st.session_state.config
        age_min, age_max = config['app']['age_range']
        default_age = config['app']['default_age']
        age = st.slider("Select your age:", age_min, age_max, default_age)

        predict_button = st.button("Predict Music Preference")

        if predict_button:
            make_prediction(age, gender_value, model_exists)
    else:
        st.info("Please upload a CSV file to make predictions.")


def make_prediction(age, gender_value, model_exists):
    """
    Generate and display predictions using both methods

    Args:
        age (int): User's age
        gender_value (int): User's gender (1 for Male, 0 for Female)
        model_exists (bool): Whether a usable model exists
    """
    can_predict = True

    # Check if we can use ML model
    can_use_ml = st.session_state.get('model_trained', False) or model_exists

    # We always have current data at this point
    can_use_percentage = True
    data = st.session_state['current_data']

    if not can_use_ml and not can_use_percentage:
        st.warning("No model or data available for prediction.")
        can_predict = False

    if can_predict:
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        # Percentage-based prediction
        with col1:
            st.info("Based on Percentage Model")
            percentage_result = predict_using_percentage(data, age, gender_value)
            display_percentage_results(percentage_result)

        # ML-based prediction
        with col2:
            display_ml_prediction(data, age, gender_value, can_use_ml)

        logger.info(f"Predictions displayed for age={age}, gender={gender_value}")


def display_ml_prediction(data, age, gender_value, can_use_ml):
    """
    Display the ML-based prediction results

    Args:
        data (pd.DataFrame): Dataset
        age (int): User's age
        gender_value (int): User's gender (1 for Male, 0 for Female)
        can_use_ml (bool): Whether ML prediction is available
    """
    st.info("Based on Machine Learning Model")

    if not can_use_ml:
        st.write("No ML model available for prediction")
        return

    try:
        if st.session_state.get('trained_model') is not None and st.session_state.get('label_encoder') is not None:
            # Use the model from session state
            ml_result = predict_using_ml(
                None,
                age,
                gender_value,
                st.session_state['trained_model'],
                st.session_state['label_encoder']
            )
        else:
            # Load the saved model
            ml_result = predict_using_ml(data, age, gender_value)

        st.write(f"**Predicted Genre:** {ml_result['genre']}")
        st.write(f"**Confidence:** {ml_result['confidence']:.2f}%")

        # Display probabilities if available
        if 'probabilities' in ml_result and ml_result['probabilities']:
            show_probability_chart(ml_result['probabilities'])
    except Exception as e:
        st.error(f"Error making ML prediction: {str(e)}")
        logger.error(f"Error making ML prediction: {str(e)}", exc_info=True)


def show_probability_chart(probabilities):
    """
    Create and display a bar chart of probabilities

    Args:
        probabilities (dict): Dictionary of genre probabilities
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    genres = list(probabilities.keys())
    probs = [probabilities[g] * 100 for g in genres]

    # Sort by probability (descending)
    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    sorted_genres = [genres[i] for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]

    # Plot bars
    bars = ax.bar(sorted_genres, sorted_probs, color='lightgreen')

    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')

    ax.set_ylim(0, max(probs) * 1.1)
    ax.set_ylabel('Probability (%)')
    ax.set_title('Genre Probabilities from ML Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(fig)