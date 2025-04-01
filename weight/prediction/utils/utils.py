import os
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
mean_squared_error,r2_score, mean_absolute_error
)

from sklearn.ensemble import RandomForestClassifier

from matplotlib.backends.backend_pdf import PdfPages
import logging

logger = logging.getLogger(__name__)


def generate_training_report(
    model,  # Trained regression model (e.g., LinearRegression)
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    timestamp: str
) -> Dict[str, Any]:
    """
    Generate a comprehensive training report for a regression model with metrics and visualizations,
    and save a PDF report including key plots and metrics.

    Args:
        model: Trained regression model (e.g., LinearRegression)
        X_train: Training features
        X_test: Testing features
        y_train: Training target values
        y_test: Testing target values
        timestamp: Timestamp string for naming the report

    Returns:
        Dict: Report metrics and file paths for generated plots and reports
    """
    # Create directory for reports if it doesn't exist
    config = st.session_state.config
    models_dir = config['model']['models_dir']
    reports_dir = os.path.join(models_dir, 'train_reports')
    os.makedirs(reports_dir, exist_ok=True)

    # Base path for report files
    base_path = os.path.join(reports_dir, f'training_report_{timestamp}')

    # Get predictions on test set
    y_pred = model.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Generate scatter plot: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    scatter_plot_path = f"{base_path}_scatter.png"
    plt.savefig(scatter_plot_path, bbox_inches='tight')
    plt.close()

    # Generate residual plot: Residuals vs Predicted
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    residual_plot_path = f"{base_path}_residuals.png"
    plt.savefig(residual_plot_path, bbox_inches='tight')
    plt.close()

    # Generate coefficient plot if the model exposes coefficients
    if hasattr(model, 'coef_'):
        # Attempt to get feature names from config, or generate default names
        feature_names = config.get('feature_names', [f'feature_{i}' for i in range(X_train.shape[1])])
        coefs = model.coef_
        # Handle multi-output regression by selecting the first set of coefficients if needed
        if coefs.ndim > 1:
            coefs = coefs[0]
        indices = np.argsort(np.abs(coefs))[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Feature Coefficients')
        plt.bar(range(len(coefs)), coefs[indices], align='center')
        plt.xticks(range(len(coefs)), ["Height", "Gender"], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        coef_plot_path = f"{base_path}_coefficients.png"
        plt.savefig(coef_plot_path, bbox_inches='tight')
        plt.close()
    else:
        coef_plot_path = None

    # Generate PDF report
    pdf_path = f"{base_path}.pdf"
    with PdfPages(pdf_path) as pdf:
        # Title page
        plt.figure(figsize=(11.7, 8.3))
        plt.text(0.5, 0.5, f"Regression Model Training Report\n\n{timestamp}",
                 horizontalalignment='center', verticalalignment='center', fontsize=20)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # Scatter Plot page
        scatter_img = plt.imread(scatter_plot_path)
        plt.figure(figsize=(11.7, 8.3))
        plt.imshow(scatter_img)
        plt.axis('off')
        plt.title('Actual vs Predicted Values')
        pdf.savefig()
        plt.close()

        # Residual Plot page
        residual_img = plt.imread(residual_plot_path)
        plt.figure(figsize=(11.7, 8.3))
        plt.imshow(residual_img)
        plt.axis('off')
        plt.title('Residual Plot')
        pdf.savefig()
        plt.close()

        # Coefficient Plot page (if available)
        if coef_plot_path is not None:
            coef_img = plt.imread(coef_plot_path)
            plt.figure(figsize=(11.7, 8.3))
            plt.imshow(coef_img)
            plt.axis('off')
            plt.title('Feature Coefficients')
            pdf.savefig()
            plt.close()

        # Metrics page
        plt.figure(figsize=(11.7, 8.3))
        plt.axis('off')
        plt.text(0.1, 0.9, "Model Performance Metrics", fontsize=18)
        metrics_text = [
            f"MSE: {mse:.4f}",
            f"MAE: {mae:.4f}",
            f"R²: {r2:.4f}"
        ]
        plt.text(0.1, 0.8, "\n".join(metrics_text), fontsize=12)
        pdf.savefig()
        plt.close()

    logger.info(f"PDF report generated at: {pdf_path}")

    # Save metrics to a text file
    metrics_path = f"{base_path}_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("# Regression Model Training Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Model Information\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        if hasattr(model, 'coef_'):
            f.write("Model has coefficients.\n")
        f.write("\n## Dataset Information\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n\n")
        f.write("## Performance Metrics\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R²: {r2:.4f}\n")

    # Save report data as JSON for potential future use
    report_data = {
        "timestamp": timestamp,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "model_type": type(model).__name__,
        "training_samples": len(X_train),
        "testing_samples": len(X_test)
    }

    json_path = f"{base_path}_report.json"
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Training report generated: {metrics_path}")

    return {
        "metrics_path": metrics_path,
        "scatter_plot_path": scatter_plot_path,
        "residual_plot_path": residual_plot_path,
        "coef_plot_path": coef_plot_path,
        "pdf_path": pdf_path,
        "json_path": json_path,
        "mse": mse,
        "mae": mae,
        "r2": r2
    }