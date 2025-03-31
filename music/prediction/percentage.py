"""
Percentage-based prediction module for music preferences.
"""
import pandas as pd
import streamlit as st
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def determine_age_group(age: int) -> str:
    """
    Determine the age group for a given age based on configuration.

    Args:
        age (int): Age value

    Returns:
        str: Age group label
    """
    config = st.session_state.config
    bins = config['data']['age_bins']
    labels = config['data']['age_labels']

    for i in range(len(bins) - 1):
        if bins[i] <= age < bins[i + 1]:
            return labels[i]

    # Default to the last group if no match
    return labels[-1]

def predict_using_percentage(data: pd.DataFrame, age: int, gender: int) -> Dict[str, Any]:
    """
    Predict music genre preference using percentage-based statistics.
    Returns all genre percentages, not just the top one.

    Args:
        data (pd.DataFrame): Dataset containing age, gender, and genre
        age (int): User's age
        gender (int): User's gender (1 for male, 0 for female)

    Returns:
        Dict[str, Any]: Dictionary with predicted genre, confidence, and all percentages
    """
    try:
        # Determine age group
        age_group = determine_age_group(age)
        logger.info(f"Determined age group '{age_group}' for age {age}")

        # Filter by gender
        filtered_data = data[data['gender'] == gender]
        logger.info(f"Filtered by gender {gender}: {len(filtered_data)} rows")

        # Filter by age group according to the configuration
        config = st.session_state.config
        bins = config['data']['age_bins']

        # Extract the age thresholds for the current age group
        age_group_index = config['data']['age_labels'].index(age_group)
        min_age = bins[age_group_index]
        max_age = bins[age_group_index + 1]

        # Apply filter based on these thresholds
        if age_group_index == len(config['data']['age_labels']) - 1:  # Last group
            filtered_data = filtered_data[filtered_data['age'] >= min_age]
        else:
            filtered_data = filtered_data[(filtered_data['age'] >= min_age) & (filtered_data['age'] < max_age)]

        logger.info(f"Filtered by age group '{age_group}': {len(filtered_data)} rows")

        # Determine the genre column name (could be 'genre' or 'music_preference')
        genre_column = 'genre' if 'genre' in filtered_data.columns else 'music_preference'

        # Count genres
        if len(filtered_data) > 0:
            genre_counts = filtered_data[genre_column].value_counts()
            total_count = len(filtered_data)

            # Calculate percentages for all genres
            all_percentages = {}
            for genre, count in genre_counts.items():
                percentage = (count / total_count) * 100
                all_percentages[genre] = percentage

            # Sort percentages by value (descending)
            sorted_percentages = {k: v for k, v in sorted(
                all_percentages.items(),
                key=lambda item: item[1],
                reverse=True
            )}

            # Get the most common genre (top one)
            top_genre = list(sorted_percentages.keys())[0]
            top_confidence = sorted_percentages[top_genre]

            logger.info(f"Percentage prediction: {top_genre}, Confidence: {top_confidence:.2f}%")
            logger.info(f"All percentages: {sorted_percentages}")

            return {
                "genre": top_genre,
                "confidence": top_confidence,
                "all_percentages": sorted_percentages
            }
        else:
            logger.warning("No matching data for percentage prediction")
            return {
                "genre": "Unknown",
                "confidence": 0,
                "all_percentages": {}
            }
    except Exception as e:
        logger.error(f"Error in percentage prediction: {str(e)}", exc_info=True)
        return {
            "genre": f"Error: {str(e)}",
            "confidence": 0,
            "all_percentages": {}
        }