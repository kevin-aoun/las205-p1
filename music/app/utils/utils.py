import streamlit as st
import pandas as pd

def display_percentage_results(percentage_result):
    """
    Display all the percentages from the percentage-based prediction.

    Args:
        percentage_result (dict): Result from predict_using_percentage
    """
    st.write(f"**Predicted Genre:** {percentage_result['genre']}")
    st.write(f"**Top Confidence:** {percentage_result['confidence']:.2f}%")

    # If we have all percentages, display them as a table
    if 'all_percentages' in percentage_result and percentage_result['all_percentages']:
        st.write("**All Genre Percentages:**")

        # Get the percentages
        genres = list(percentage_result['all_percentages'].keys())
        percentages = list(percentage_result['all_percentages'].values())

        # Sort by percentage (descending)
        sorted_indices = sorted(range(len(percentages)), key=lambda i: percentages[i], reverse=True)
        sorted_genres = [genres[i] for i in sorted_indices]
        sorted_percentages = [percentages[i] for i in sorted_indices]

        # Create and display the table
        percentage_df = pd.DataFrame({
            'Genre': sorted_genres,
            'Percentage': [f"{p:.2f}%" for p in sorted_percentages]
        })
        st.dataframe(percentage_df, use_container_width=True)