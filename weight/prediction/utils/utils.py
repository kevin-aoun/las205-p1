import os
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt
import json
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error
)

from matplotlib.backends.backend_pdf import PdfPages
from weight.logs import logger


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

    # Get predictions on both train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    # Generate enhanced scatter plot: Actual vs Predicted for both train and test
    plt.figure(figsize=(10, 8))

    # Plot training data points (circles)
    plt.scatter(y_train, y_train_pred, color='blue', marker='o', alpha=0.6, label='Training Data')

    # Plot test data points (triangles)
    plt.scatter(y_test, y_test_pred, color='red', marker='^', alpha=0.8, label='Test Data')

    # Add the perfect prediction line
    all_y = np.concatenate([y_train, y_test])
    min_val, max_val = min(all_y), max(all_y)
    plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle='--', label='Perfect Prediction')

    plt.xlabel('Actual Weight', fontsize=12)
    plt.ylabel('Predicted Weight', fontsize=12)
    plt.title('Actual vs. Predicted Weight', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    # Add annotations showing train and test sizes
    plt.annotate(f'Train set: {len(y_train)} samples',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    plt.annotate(f'Test set: {len(y_test)} samples',
                 xy=(0.05, 0.90), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    scatter_plot_path = f"{base_path}_scatter.png"
    plt.savefig(scatter_plot_path, bbox_inches='tight', dpi=100)
    plt.close()

    # Generate residual plot: Residuals vs Predicted
    residuals = y_test - y_test_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    residual_plot_path = f"{base_path}_residuals.png"
    plt.savefig(residual_plot_path, bbox_inches='tight')
    plt.close()

    # Generate Weight vs Height by Gender plot
    # Combine train and test data for a comprehensive view
    X_all = np.vstack((X_train, X_test))
    y_all = np.concatenate((y_train, y_test))

    # Assuming X has height in the first column and gender in the second column
    heights = X_all[:, 0]
    genders = X_all[:, 1]  # Assuming 0 for female, 1 for male or similar encoding
    weights = y_all

    plt.figure(figsize=(10, 8))

    # Separate data by gender
    female_indices = genders == 0
    male_indices = genders == 1

    # Plot each gender with different colors
    plt.scatter(heights[female_indices], weights[female_indices],
                color='pink', marker='o', alpha=0.7, label='Female')
    plt.scatter(heights[male_indices], weights[male_indices],
                color='blue', marker='o', alpha=0.7, label='Male')

    plt.xlabel('Height', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.title('Weight vs Height by Gender', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    # Add trend lines for each gender
    if np.any(female_indices):
        z = np.polyfit(heights[female_indices], weights[female_indices], 1)
        p = np.poly1d(z)
        plt.plot(heights[female_indices], p(heights[female_indices]),
                 "r--", color='deeppink', alpha=0.8)

    if np.any(male_indices):
        z = np.polyfit(heights[male_indices], weights[male_indices], 1)
        p = np.poly1d(z)
        plt.plot(heights[male_indices], p(heights[male_indices]),
                 "r--", color='darkblue', alpha=0.8)

    gender_plot_path = f"{base_path}_gender_height_weight.png"
    plt.savefig(gender_plot_path, bbox_inches='tight', dpi=100)
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
        plt.title('Actual vs Predicted Values (Blue: Training, Red: Test)')
        pdf.savefig()
        plt.close()

        # Gender Height Weight Plot page
        gender_img = plt.imread(gender_plot_path)
        plt.figure(figsize=(11.7, 8.3))
        plt.imshow(gender_img)
        plt.axis('off')
        plt.title('Weight vs Height by Gender')
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

        # Calculate train metrics as well
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        metrics_text = [
            f"Training Set Metrics:",
            f"  - MSE: {train_mse:.4f}",
            f"  - MAE: {train_mae:.4f}",
            f"  - R²: {train_r2:.4f}",
            f"\nTest Set Metrics:",
            f"  - MSE: {mse:.4f}",
            f"  - MAE: {mae:.4f}",
            f"  - R²: {r2:.4f}",
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
        f.write("Training Set:\n")
        f.write(f"  MSE: {train_mse:.4f}\n")
        f.write(f"  MAE: {train_mae:.4f}\n")
        f.write(f"  R²: {train_r2:.4f}\n\n")
        f.write("Test Set:\n")
        f.write(f"  MSE: {mse:.4f}\n")
        f.write(f"  MAE: {mae:.4f}\n")
        f.write(f"  R²: {r2:.4f}\n")

    # Save report data as JSON for potential future use
    report_data = {
        "timestamp": timestamp,
        "test_metrics": {
            "mse": mse,
            "mae": mae,
            "r2": r2
        },
        "train_metrics": {
            "mse": train_mse,
            "mae": train_mae,
            "r2": train_r2
        },
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
        "gender_height_weight_plot_path": gender_plot_path,
        "coef_plot_path": coef_plot_path,
        "pdf_path": pdf_path,
        "json_path": json_path,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "train_mse": train_mse,
        "train_mae": train_mae,
        "train_r2": train_r2
    }