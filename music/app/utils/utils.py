import streamlit as st
import pandas as pd

def display_prediction_result(result):
    """Display prediction results in a consistent format for both methods"""
    st.write(f"**Predicted Genre:** {result['genre']}")
    st.write(f"**Confidence:** {result['confidence']:.2f}%")

    # Display all percentages/probabilities if available
    percentages = result.get('all_percentages') or result.get('probabilities')
    if percentages:
        st.write("**All Percentages:**")

        # Get values and sort
        labels = list(percentages.keys())
        values = [percentages[label] * 100 if percentages[label] <= 1 else percentages[label] for label in labels]

        # Sort by value (descending)
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]

        # Create and display the table
        data_df = pd.DataFrame({
            'Genre': sorted_labels,
            'Percentage': [f"{v:.2f}%" for v in sorted_values]
        })
        st.dataframe(data_df, use_container_width=True)