import os
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages
import logging
from PIL import Image

logger = logging.getLogger(__name__)

def create_plot_figure(figsize: tuple = (8, 6)) -> plt.Figure:
    """Create and return a new matplotlib figure."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def save_and_close_plot(fig: plt.Figure, filepath: str) -> None:
    """Save the plot to the specified path and close the figure."""
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

def generate_scatter_plot(X: np.ndarray, y: np.ndarray, feature_idx: int, title: str, xlabel: str, ylabel: str, filepath: str):
    """Generate and save a scatter plot."""
    fig, ax = create_plot_figure()
    ax.scatter(X[:, feature_idx], y, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_and_close_plot(fig, filepath)

def generate_actual_vs_predicted_plot(y_pred: np.ndarray, y_test: np.ndarray, title: str, xlabel: str, ylabel: str, filepath: str):
    """Generate and save a scatter plot."""
    fig, ax = create_plot_figure()
    ax.scatter(y_pred,y_test, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_and_close_plot(fig, filepath)

def generate_best_fit_line(X: np.ndarray, y: np.ndarray, feature_idx: int, filepath: str):
    """Generate and save a scatter plot with a line of best fit."""
    fig, ax = create_plot_figure()
    ax.scatter(X[:, feature_idx], y, alpha=0.7, color='blue')

    # Fit a line of best fit (linear regression)
    slope, intercept = np.polyfit(X[:, feature_idx], y, 1)
    ax.plot(X[:, feature_idx], slope * X[:, feature_idx] + intercept, color='red', linestyle='--')
    
    ax.set_xlabel('Height')
    ax.set_ylabel('Weight Kg')
    ax.set_title('Height vs Weight with Line of Best Fit')
    save_and_close_plot(fig, filepath)

def generate_residual_plot(y_test: np.ndarray, y_pred: np.ndarray, filepath: str):
    """Generate and save a residual plot."""
    residuals = y_test - y_pred
    fig, ax = create_plot_figure()
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    save_and_close_plot(fig, filepath)

def generate_training_report(
        model, X_train, X_test, y_train, y_test, timestamp: str
) -> Dict[str, Any]:
    """Generate a comprehensive training report with metrics and visualizations."""
    try:
        # Create directory for reports if it doesn't exist
        config = st.session_state.config
        models_dir = config['model']['models_dir']
        reports_dir = os.path.join(models_dir, 'train_reports')
        os.makedirs(reports_dir, exist_ok=True)

        # Base path for report files
        base_path = os.path.join(reports_dir, f'training_report_{timestamp}')

        # Get predictions on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Generate scatter plot: Gender vs Target (y)
        gender_scatter_path = f"{base_path}_gender_scatter.png"
        generate_scatter_plot(X_train, y_train, 1, "Gender vs Weight", "Gender (0 or 1)", "Weight Kg", gender_scatter_path)

        #Compares Actual and Predicted Weight test values
        weight_comparison_path = f"{base_path}_weight_comparison.png"
        generate_actual_vs_predicted_plot(y_pred,y_test,"Actual vs. Predicted Weight","Actual Weight","Predicted Weight",weight_comparison_path)

       


        # Generate scatter plot: Height vs Target (y) with line of best fit
        height_scatter_path = f"{base_path}_height_scatter.png"
        generate_best_fit_line(X_train, y_train, 0, height_scatter_path)

        # Generate residual plot
        residual_plot_path = f"{base_path}_residuals.png"
        generate_residual_plot(y_test, y_pred, residual_plot_path)

        # Generate PDF report
        pdf_path = create_pdf_report(
            base_path, timestamp, model, X_train, X_test, mse, mae, r2,
            gender_scatter_path, height_scatter_path, residual_plot_path
        )

        # Save metrics to a text file
        metrics_path = save_metrics_text(
            base_path, timestamp, model, X_train, X_test, mse, mae, r2
        )

        # Save report data as JSON for potential future use
        json_path = save_json_report(
            base_path, timestamp, mse, mae, r2, X_test, X_train
        )

        logger.info(f"Training report generated successfully: {metrics_path}")

        return {
            "metrics_path": metrics_path,
            
            "pdf_path": pdf_path,
            "json_path": json_path,
            "mse": mse,
            "mae": mae,
            "r2": r2
        }

    except Exception as e:
        logger.error(f"Failed to generate training report: {str(e)}", exc_info=True)
        return {}

def create_pdf_report(base_path, timestamp, model, X_train, X_test, mse, mae, r2,
                      gender_scatter_path, height_scatter_path, residual_plot_path) -> str:
    """Create and save a PDF report with all metrics and visualizations."""
    try:
        pdf_path = f"{base_path}.pdf"
        with PdfPages(pdf_path) as pdf:
            # Title page
            plt.figure(figsize=(11.7, 8.3))
            plt.text(0.5, 0.5, f"Model Training Report\n\n{timestamp}",
                     horizontalalignment='center', verticalalignment='center', fontsize=20)
            plt.axis('off')
            pdf.savefig()
            plt.close()

            # Gender vs Weight plot
            gender_scatter_img = Image.open(gender_scatter_path)
            plt.figure(figsize=(11.7, 8.3))
            plt.imshow(gender_scatter_img)
            plt.axis('off')
            plt.title('Gender vs Weight')
            pdf.savefig()
            plt.close()

            # Height vs Weight plot with line of best fit
            height_scatter_img = Image.open(height_scatter_path)
            plt.figure(figsize=(11.7, 8.3))
            plt.imshow(height_scatter_img)
            plt.axis('off')
            plt.title('Height vs Weight with Line of Best Fit')
            pdf.savefig()
            plt.close()

            # Residual plot
            residual_img = Image.open(residual_plot_path)
            plt.figure(figsize=(11.7, 8.3))
            plt.imshow(residual_img)
            plt.axis('off')
            plt.title('Residual Plot')
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
        return pdf_path

    except Exception as e:
        logger.error(f"Failed to create PDF report: {str(e)}", exc_info=True)
        return ""

def save_metrics_text(base_path, timestamp, model, X_train, X_test, mse, mae, r2) -> str:
    """Save metrics to a text file."""
    try:
        metrics_path = f"{base_path}_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("# Model Training Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Model Information\n")
            f.write(f"Model Type: Linear Regression\n")
          

            f.write("## Dataset Information\n")
            f.write(f"Training samples: {len(X_train)}\n")
            f.write(f"Testing samples: {len(X_test)}\n\n")

            f.write("## Performance Metrics\n")
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"R²: {r2:.4f}\n")

        return metrics_path

    except Exception as e:
        logger.error(f"Failed to save metrics text: {str(e)}", exc_info=True)
        return ""

def save_json_report(base_path, timestamp, mse, mae, r2, X_test, X_train) -> str:
    """Save report data as JSON."""
    try:
        report_data = {
            "timestamp": timestamp,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "model_type": "Linear Regression",
            "training_samples": len(X_train),
            "testing_samples": len(X_test)

        }

        json_path = f"{base_path}_report.json"
        import json
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        return json_path

    except Exception as e:
        logger.error(f"Failed to save JSON report: {str(e)}", exc_info=True)
        return ""
