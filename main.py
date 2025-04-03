from weight.app import run_app2
from music.app import run_app
import streamlit as st


# def render_selector_tab():
#     """
#     Render a tab that allows switching between Weight and Genre predictors.
#     """
#     st.title("🧠 Smart Predictor")
#     st.write("Choose a prediction mode below to get started:")
#
#     predictor_choice = st.radio(
#         "Select Predictor Type:",
#         options=[ "Genre Predictor", "Weight Predictor"],
#         index=0,
#         horizontal=True
#     )
#
#     st.markdown("---")
#     if predictor_choice == "Genre Predictor":
#         st.header("🎵 Predict Music Genre")
#         st.write("Predict music genre preference based on demographic data (classification).")
#         run_app()
#     elif predictor_choice == "Weight Predictor":
#         st.header("🏋️ Predict Weight")
#         st.write("Use your height and gender to predict weight (numeric regression).")
#
#         run_app2()

    

if __name__ == "__main__":
    run_app()

