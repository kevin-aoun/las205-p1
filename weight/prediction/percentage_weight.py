"""
Percentage-based prediction module for weight categories.
"""
import pandas as pd
import streamlit as st
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def determine_age_group(Gender: str) -> str:
    """
    Determine the gender group (replacing age group in original logic).

    Args:
        Gender (str): Gender value (e.g., 'Male', 'Female')

    Returns:
        str: Gender group label
    """
    return Gender  # assuming direct mapping, no bins needed

def predict_using_percentage(data: pd.DataFrame, Gender: str, Height: float) -> Dict[str, Any]:
    """
    Predict Weight category using percentage-based statistics.
    Returns all Weight percentages, not just the top one.

    Args:
        data (pd.DataFrame): Dataset containing Gender, Height, and Weight
        Gender (str): User's gender (e.g., 'Male', 'Female')
        Height (float): User's height in cm

    Returns:
        Dict[str, Any]: Dictionary with predicted Weight, confidence, and all percentages
    """
    try:
        # Determine gender group (reuse determine_age_group just for structure)
        gender_group = determine_age_group(Gender)
        logger.info(f"Determined gender group '{gender_group}' for Gender {Gender}")

        # Filter by Gender
        filtered_data = data[data['Gender'] == gender_group]
        logger.info(f"Filtered by Gender {Gender}: {len(filtered_data)} rows")

        # Filter by Height group using config
        config = st.session_state.config
        bins = config['data']['height_bins']
        labels = config['data']['height_labels']

        # Find height group
        height_group_index = None
        for i in range(len(bins) - 1):
            if bins[i] <= Height < bins[i + 1]:
                height_group_index = i
                break
        if height_group_index is None:
            height_group_index = len(bins) - 2

        min_height = bins[height_group_index]
        max_height = bins[height_group_index + 1]

        # Apply height filter
        if height_group_index == len(labels) - 1:
            filtered_data = filtered_data[filtered_data['Height'] >= min_height]
        else:
            filtered_data = filtered_data[(filtered_data['Height'] >= min_height) & (filtered_data['Height'] < max_height)]

        logger.info(f"Filtered by height group '{labels[height_group_index]}': {len(filtered_data)} rows")

        # Determine the weight column name
        weight_column = 'Weight' if 'Weight' in filtered_data.columns else 'weight_category'

        # Count weights
        if len(filtered_data) > 0:
            weight_counts = filtered_data[weight_column].value_counts()
            total_count = len(filtered_data)

            # Calculate percentages
            all_percentages = {
                weight: (count / total_count) * 100
                for weight, count in weight_counts.items()
            }

            sorted_percentages = dict(sorted(all_percentages.items(), key=lambda item: item[1], reverse=True))

            top_weight = list(sorted_percentages.keys())[0]
            top_confidence = sorted_percentages[top_weight]

            logger.info(f"Percentage prediction: {top_weight}, Confidence: {top_confidence:.2f}%")
            logger.info(f"All percentages: {sorted_percentages}")

            return {
                "Weight": top_weight,  # keeping original key for output
                "confidence": top_confidence,
                "all_percentages": sorted_percentages
            }
        else:
            logger.warning("No matching data for percentage prediction")
            return {
                "Weight": "Unknown",
                "confidence": 0,
                "all_percentages": {}
            }

    except Exception as e:
        logger.error(f"Error in percentage prediction: {str(e)}", exc_info=True)
        return {
            "Weight": f"Error: {str(e)}",
            "confidence": 0,
            "all_percentages": {}
        }
