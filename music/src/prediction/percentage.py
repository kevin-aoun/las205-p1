import logging

logger = logging.getLogger(__name__)

def predict_using_percentage(data, age, gender):
    """
    Predict music genre preference using percentage-based statistics.

    Args:
        data (pd.DataFrame): Dataset containing age, gender, and genre
        age (int): User's age
        gender (int): User's gender (1 for male, 0 for female)

    Returns:
        dict: Dictionary with predicted genre and confidence
    """
    # Define age groups
    if age <= 25:
        age_group = "young"
    elif age <= 31:
        age_group = "mid"
    else:
        age_group = "older"

    filtered_data = data[data['gender'] == gender]

    if age_group == "young":
        filtered_data = filtered_data[filtered_data['age'] <= 25]
    elif age_group == "mid":
        filtered_data = filtered_data[(filtered_data['age'] > 25) & (filtered_data['age'] <= 31)]
    else:
        filtered_data = filtered_data[filtered_data['age'] > 31]

    # Count genres
    if len(filtered_data) > 0:
        genre_counts = filtered_data['genre'].value_counts()
        total_count = len(filtered_data)

        # Get the most common genre
        top_genre = genre_counts.index[0]
        confidence = (genre_counts[top_genre] / total_count) * 100

        logger.info(f"Percentage predictions: {top_genre}, Confidence: {confidence}")

        return {
            "genre": top_genre,
            "confidence": confidence
        }
    else:
        return {
            "genre": "Unknown",
            "confidence": 0
        }