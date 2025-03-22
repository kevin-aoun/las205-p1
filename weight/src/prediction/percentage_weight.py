import logging

logger = logging.getLogger(__name__)

def predict_weight_percentage(data, gender, height):
    """
    Estimate weight using basic percentage/statistical method.

    Args:
        data (DataFrame): Dataset with 'height', 'gender', and 'weight'
        height (float): User height
        gender (int): 1 for male, 0 for female

    Returns:
        dict: Predicted weight and confidence
    """
    filtered = data[data['Gender'] == gender]

    closest = filtered.iloc[(filtered['Height'] - height).abs().argsort()[:10]]

    if len(closest) > 0:
        avg_weight = closest['Weight'].mean()
        logger.info(f"Predicted (percentage-based): {avg_weight:.2f}kg")
        return {"weight": round(avg_weight, 2), "confidence": 85.0}
    else:
        return {"weight": "Unknown", "confidence": 0}
