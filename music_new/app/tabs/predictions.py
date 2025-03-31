import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from music_new.core import logger
from music_new.prediction import predict_using_percentage, predict_using_ml
from music_new.app.utils import display_percentage_results


def render_prediction_tab(model_exists):
    """
    Render the Prediction tab

    Args:
        model_exists (bool): Whether a usable model exists
    """
    if st.session_state.get('current_data') is not None:
        st.subheader("Enter Your Information")

        gender = st.radio("Select your gender:",
                          options=["Male", "Female"],
                          index=0)
        gender_value = 1 if gender == "Male" else 0

        config = st.session_state.config
        height_min, height_max = config['app']['height_range']
        default_height = config['app']['default_height']
        height = st.slider("Select your height:", height_min, height_max, default_height)

        predict_button = st.button("Predict Weight")

        if predict_button:
            make_prediction(height, gender_value, model_exists)
    else:
        st.info("Please upload a CSV file to make predictions.")


def make_prediction(height, gender_value, model_exists):
    """
    Generate and display predictions using both methods

    Args:
        height (float): User's height
        gender_value (int): User's gender (1 for Male, 0 for Female)
        model_exists (bool): Whether a usable model exists
    """
    can_predict = True
    can_use_ml = st.session_state.get('model_trained', False) or model_exists
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
            percentage_result = predict_using_percentage(data, gender_value, height)
            display_percentage_results(percentage_result)

        # ML-based prediction
        with col2:
            display_ml_prediction(data, height, gender_value, can_use_ml)

        logger.info(f"Predictions displayed for height={height}, gender={gender_value}")


def display_ml_prediction(data, height, gender_value, can_use_ml):
    """
    Display the ML-based prediction results

    Args:
        data (pd.DataFrame): Dataset
        height (float): User's height
        gender_value (int): User's gender (1 for Male, 0 for Female)
        can_use_ml (bool): Whether ML prediction is available
    """
    st.info("Based on Machine Learning Model")

    if not can_use_ml:
        st.write("No ML model available for prediction")
        return

    try:
        if st.session_state.get('trained_model') is not None:
            ml_result = predict_using_ml(
                data=None,
                Gender="Male" if gender_value == 1 else "Female",
                Height=height,
                trained_model=st.session_state['trained_model']
            )
        else:
            ml_result = predict_using_ml(
                data=data,
                Gender="Male" if gender_value == 1 else "Female",
                Height=height
            )

        st.write(f"**Predicted Weight:** {ml_result['Weight']}")
        if ml_result['confidence'] is not None:
            st.write(f"**Confidence:** {ml_result['confidence']:.2f}%")

        # Optional: Display probabilities if applicable
        if 'probabilities' in ml_result and ml_result['probabilities']:
            show_probability_chart(ml_result['probabilities'])

    except Exception as e:
        st.error(f"Error making ML prediction: {str(e)}")
        logger.error(f"Error making ML prediction: {str(e)}", exc_info=True)


def show_probability_chart(probabilities):
    """
    Create and display a bar chart of weight category probabilities

    Args:
        probabilities (dict): Dictionary of weight probabilities
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    weight_classes = list(probabilities.keys())
    probs = [probabilities[w] * 100 for w in weight_classes]

    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    sorted_labels = [weight_classes[i] for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]

    bars = ax.bar(sorted_labels, sorted_probs, color='lightgreen')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')

    ax.set_ylim(0, max(sorted_probs) * 1.1)
    ax.set_ylabel('Probability (%)')
    ax.set_title('Weight Probabilities from ML Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(fig)
