import streamlit as st
import pandas as pd

def display_percentage_results(percentage_result):
    """
    Display all the percentages from the percentage-based prediction.

    Args:
        percentage_result (dict): Result from predict_using_percentage
    """
    st.write(f"**Predicted Weight:** {percentage_result['Weight']}")
    st.write(f"**Top Confidence:** {percentage_result['confidence']:.2f}%")

    # If we have all percentages, display them as a table
    if 'all_percentages' in percentage_result and percentage_result['all_percentages']:
        st.write("**All Weight Percentages:**")

        # Get the percentages
        weights = list(percentage_result['all_percentages'].keys())
        percentages = list(percentage_result['all_percentages'].values())

        # Sort by percentage (descending)
        sorted_indices = sorted(range(len(percentages)), key=lambda i: percentages[i], reverse=True)
        sorted_weights = [weights[i] for i in sorted_indices]
        sorted_percentages = [percentages[i] for i in sorted_indices]

        # Create and display the table
        percentage_df = pd.DataFrame({
            'Weight': sorted_weights,
            'Percentage': [f"{p:.2f}%" for p in sorted_percentages]
        })
        st.dataframe(percentage_df, use_container_width=True)
