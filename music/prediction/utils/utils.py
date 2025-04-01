import os
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.tree import export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib.backends.backend_pdf import PdfPages
from music.logs import logger

def export_tree_visualization(model, feature_names, class_names, timestamp):
    """
    Generate and save a visualization of a tree from the Random Forest.

    Args:
        model: Trained RandomForestClassifier model
        feature_names: List of feature names
        class_names: List of class names
        timestamp: Timestamp string for naming the output file

    Returns:
        tuple: (tree_png_path, tree_dot_path) or (None, None) if failed
    """
    try:
        config = st.session_state.config
        models_dir = config['model']['models_dir']
        reports_dir = os.path.join(models_dir, 'train_reports')
        os.makedirs(reports_dir, exist_ok=True)

        # Sample one tree from the forest (the first one)
        estimator = model.estimators_[0]

        # Path for PNG output
        tree_png_path = os.path.join(reports_dir, f'tree_viz_{timestamp}.png')

        # Create the visualization using plot_tree
        plt.figure(figsize=(20, 10))
        plot_tree(
            estimator,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.savefig(tree_png_path, bbox_inches='tight', dpi=150)
        plt.close()

        # Also create a DOT file visualization
        dot_path = os.path.join(reports_dir, f'tree_viz_{timestamp}.dot')
        export_graphviz(
            estimator,
            out_file=dot_path,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True
        )

        logger.info(f"Tree visualization saved to: {tree_png_path}")
        return tree_png_path, dot_path

    except Exception as e:
        logger.error(f"Failed to export tree visualization: {str(e)}", exc_info=True)
        return None, None


def generate_training_report(
        model: RandomForestClassifier,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        le: LabelEncoder,
        timestamp: str
) -> Dict[str, Any]:
    """
    Generate a comprehensive training report with metrics and visualizations.

    Args:
        model: Trained RandomForestClassifier model
        X_train: Training features
        X_test: Testing features
        y_train: Training labels (encoded)
        y_test: Testing labels (encoded)
        le: Label encoder for decoding predictions
        timestamp: Timestamp string for naming the report

    Returns:
        Dict: Report metrics and file paths or empty dict if failed
    """
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
        accuracy = accuracy_score(y_test, y_pred)
        precision_micro = precision_score(y_test, y_pred, average='micro')
        precision_macro = precision_score(y_test, y_pred, average='macro')
        precision_weighted = precision_score(y_test, y_pred, average='weighted')

        recall_micro = recall_score(y_test, y_pred, average='micro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')

        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        # Get class labels
        classes = le.classes_

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Generate confusion matrix heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Save confusion matrix plot
        cm_path = f"{base_path}_confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()

        # Generate feature importance plot
        feature_names = ['age', 'gender']
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlim([-1, len(importances)])

        # Save feature importance plot
        fi_path = f"{base_path}_feature_importance.png"
        plt.savefig(fi_path, bbox_inches='tight')
        plt.close()

        # Get detailed classification report (avoid target_names to prevent mismatch)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Create tree visualization
        tree_png_path, tree_dot_path = export_tree_visualization(
            model, feature_names, classes, timestamp
        )

        # Generate PDF report with improved layout
        pdf_path = create_pdf_report(
            base_path, timestamp, model, X_train, X_test,
            classes, cm, importances, indices, feature_names,
            accuracy, precision_micro, precision_macro, precision_weighted,
            recall_micro, recall_macro, recall_weighted,
            f1_micro, f1_macro, f1_weighted,
            class_report, tree_png_path
        )

        # Save metrics to a text file
        metrics_path = save_metrics_text(
            base_path, timestamp, model, X_train, X_test,
            classes, accuracy, precision_micro, precision_macro, precision_weighted,
            recall_micro, recall_macro, recall_weighted,
            f1_micro, f1_macro, f1_weighted,
            class_report
        )

        # Save report data as JSON for potential future use
        json_path = save_json_report(
            base_path, timestamp, accuracy,
            precision_micro, precision_macro, precision_weighted,
            recall_micro, recall_macro, recall_weighted,
            f1_micro, f1_macro, f1_weighted,
            class_report, cm, feature_names, importances
        )

        logger.info(f"Training report generated successfully: {metrics_path}")

        return {
            "metrics_path": metrics_path,
            "confusion_matrix_path": cm_path,
            "feature_importance_path": fi_path,
            "tree_png_path": tree_png_path,
            "tree_dot_path": tree_dot_path,
            "pdf_path": pdf_path,
            "json_path": json_path,
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted
        }

    except Exception as e:
        logger.error(f"Failed to generate training report: {str(e)}", exc_info=True)
        return {}


def create_pdf_report(
        base_path, timestamp, model, X_train, X_test,
        classes, cm, importances, indices, feature_names,
        accuracy, precision_micro, precision_macro, precision_weighted,
        recall_micro, recall_macro, recall_weighted,
        f1_micro, f1_macro, f1_weighted,
        class_report, tree_png_path
) -> str:
    """Create a PDF report with all metrics and visualizations"""
    try:
        pdf_path = f"{base_path}.pdf"
        with PdfPages(pdf_path) as pdf:
            # Title page
            plt.figure(figsize=(11.7, 8.3))
            plt.text(0.5, 0.5, f"Music Preference Model Training Report\n\n{timestamp}",
                     horizontalalignment='center', verticalalignment='center', fontsize=20)
            plt.axis('off')
            pdf.savefig()
            plt.close()

            # Confusion matrix page
            plt.figure(figsize=(11, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Feature importance page
            plt.figure(figsize=(11, 7))
            plt.title('Feature Importances')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
            plt.xlim([-1, len(importances)])
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Tree visualization page (if available)
            if tree_png_path and os.path.exists(tree_png_path):
                plt.figure(figsize=(11.7, 8.3))
                plt.title('Example Decision Tree from Forest')
                tree_img = plt.imread(tree_png_path)
                plt.imshow(tree_img)
                plt.axis('off')
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            # Overall metrics page - improved layout
            plt.figure(figsize=(11.7, 8.3))
            plt.axis('off')
            plt.text(0.1, 0.95, "Model Performance Metrics", fontsize=18)

            # Dataset info
            plt.text(0.1, 0.9, f"Dataset: {len(X_train)} training samples, {len(X_test)} test samples", fontsize=12)
            plt.text(0.1, 0.87, f"Model: RandomForestClassifier with {model.n_estimators} estimators", fontsize=12)

            # Overall accuracy
            plt.text(0.1, 0.8, f"Accuracy: {accuracy:.4f}", fontsize=14)

            # Create a table-like layout for metrics
            header_y = 0.75
            plt.text(0.1, header_y, "Metric", fontsize=12, fontweight='bold')
            plt.text(0.3, header_y, "Micro", fontsize=12, fontweight='bold')
            plt.text(0.4, header_y, "Macro", fontsize=12, fontweight='bold')
            plt.text(0.5, header_y, "Weighted", fontsize=12, fontweight='bold')

            # Draw a line under headers
            plt.axhline(y=header_y - 0.02, xmin=0.1, xmax=0.6, color='black', linewidth=1)

            # Precision row
            row_y = header_y - 0.05
            plt.text(0.1, row_y, "Precision", fontsize=12)
            plt.text(0.3, row_y, f"{precision_micro:.4f}", fontsize=12)
            plt.text(0.4, row_y, f"{precision_macro:.4f}", fontsize=12)
            plt.text(0.5, row_y, f"{precision_weighted:.4f}", fontsize=12)

            # Recall row
            row_y -= 0.04
            plt.text(0.1, row_y, "Recall", fontsize=12)
            plt.text(0.3, row_y, f"{recall_micro:.4f}", fontsize=12)
            plt.text(0.4, row_y, f"{recall_macro:.4f}", fontsize=12)
            plt.text(0.5, row_y, f"{recall_weighted:.4f}", fontsize=12)

            # F1 Score row
            row_y -= 0.04
            plt.text(0.1, row_y, "F1 Score", fontsize=12)
            plt.text(0.3, row_y, f"{f1_micro:.4f}", fontsize=12)
            plt.text(0.4, row_y, f"{f1_macro:.4f}", fontsize=12)
            plt.text(0.5, row_y, f"{f1_weighted:.4f}", fontsize=12)

            # Draw a line under metrics
            plt.axhline(y=row_y - 0.02, xmin=0.1, xmax=0.6, color='black', linewidth=1)

            # Per-class metrics table
            row_y -= 0.06
            plt.text(0.1, row_y, "Per-Class Performance", fontsize=14)

            # Class metrics header
            row_y -= 0.04
            plt.text(0.1, row_y, "Class", fontsize=12, fontweight='bold')
            plt.text(0.3, row_y, "Precision", fontsize=12, fontweight='bold')
            plt.text(0.4, row_y, "Recall", fontsize=12, fontweight='bold')
            plt.text(0.5, row_y, "F1 Score", fontsize=12, fontweight='bold')
            plt.text(0.6, row_y, "Support", fontsize=12, fontweight='bold')

            # Draw a line under headers
            plt.axhline(y=row_y - 0.02, xmin=0.1, xmax=0.7, color='black', linewidth=1)

            # Add each class
            for class_name, metrics in class_report.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue

                row_y -= 0.04
                if row_y < 0.1:  # If we're running out of room, start a new page
                    pdf.savefig()
                    plt.close()

                    # Create a new page for remaining classes
                    plt.figure(figsize=(11.7, 8.3))
                    plt.axis('off')
                    plt.text(0.1, 0.95, "Per-Class Performance (Continued)", fontsize=18)

                    # Recreate the header
                    row_y = 0.9
                    plt.text(0.1, row_y, "Class", fontsize=12, fontweight='bold')
                    plt.text(0.3, row_y, "Precision", fontsize=12, fontweight='bold')
                    plt.text(0.4, row_y, "Recall", fontsize=12, fontweight='bold')
                    plt.text(0.5, row_y, "F1 Score", fontsize=12, fontweight='bold')
                    plt.text(0.6, row_y, "Support", fontsize=12, fontweight='bold')

                    # Draw a line under headers
                    plt.axhline(y=row_y - 0.02, xmin=0.1, xmax=0.7, color='black', linewidth=1)
                    row_y -= 0.04

                plt.text(0.1, row_y, str(class_name), fontsize=12)
                plt.text(0.3, row_y, f"{metrics['precision']:.4f}", fontsize=12)
                plt.text(0.4, row_y, f"{metrics['recall']:.4f}", fontsize=12)
                plt.text(0.5, row_y, f"{metrics['f1-score']:.4f}", fontsize=12)
                plt.text(0.6, row_y, f"{metrics['support']}", fontsize=12)

            pdf.savefig()
            plt.close()

        logger.info(f"PDF report generated at: {pdf_path}")
        return pdf_path

    except Exception as e:
        logger.error(f"Failed to create PDF report: {str(e)}", exc_info=True)
        return ""


def save_metrics_text(
        base_path, timestamp, model, X_train, X_test,
        classes, accuracy, precision_micro, precision_macro, precision_weighted,
        recall_micro, recall_macro, recall_weighted,
        f1_micro, f1_macro, f1_weighted,
        class_report
) -> str:
    """Save metrics to a text file"""
    try:
        metrics_path = f"{base_path}_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("# Music Preferences Model Training Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Model Information\n")
            f.write(f"Model Type: RandomForestClassifier\n")
            f.write(f"Number of Estimators: {model.n_estimators}\n")
            f.write(f"Random State: {model.random_state}\n\n")

            f.write("## Dataset Information\n")
            f.write(f"Training samples: {len(X_train)}\n")
            f.write(f"Testing samples: {len(X_test)}\n")
            f.write(f"Number of classes: {len(classes)}\n")
            f.write(f"Classes: {', '.join(classes)}\n\n")

            f.write("## Performance Metrics\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")

            f.write("### Precision\n")
            f.write(f"Micro: {precision_micro:.4f}\n")
            f.write(f"Macro: {precision_macro:.4f}\n")
            f.write(f"Weighted: {precision_weighted:.4f}\n\n")

            f.write("### Recall\n")
            f.write(f"Micro: {recall_micro:.4f}\n")
            f.write(f"Macro: {recall_macro:.4f}\n")
            f.write(f"Weighted: {recall_weighted:.4f}\n\n")

            f.write("### F1 Score\n")
            f.write(f"Micro: {f1_micro:.4f}\n")
            f.write(f"Macro: {f1_macro:.4f}\n")
            f.write(f"Weighted: {f1_weighted:.4f}\n\n")

            f.write("## Per-Class Metrics\n")
            for class_name in classes:
                if str(class_name) in class_report:
                    metrics = class_report[str(class_name)]
                    f.write(f"### {class_name}\n")
                    f.write(f"Precision: {metrics['precision']:.4f}\n")
                    f.write(f"Recall: {metrics['recall']:.4f}\n")
                    f.write(f"F1-Score: {metrics['f1-score']:.4f}\n")
                    f.write(f"Support: {metrics['support']}\n\n")

        return metrics_path

    except Exception as e:
        logger.error(f"Failed to save metrics text: {str(e)}", exc_info=True)
        return ""


def save_json_report(
        base_path, timestamp, accuracy,
        precision_micro, precision_macro, precision_weighted,
        recall_micro, recall_macro, recall_weighted,
        f1_micro, f1_macro, f1_weighted,
        class_report, cm, feature_names, importances
) -> str:
    """Save report data as JSON"""
    try:
        report_data = {
            "timestamp": timestamp,
            "accuracy": accuracy,
            "precision": {
                "micro": precision_micro,
                "macro": precision_macro,
                "weighted": precision_weighted
            },
            "recall": {
                "micro": recall_micro,
                "macro": recall_macro,
                "weighted": recall_weighted
            },
            "f1": {
                "micro": f1_micro,
                "macro": f1_macro,
                "weighted": f1_weighted
            },
            "class_report": class_report,
            "confusion_matrix": cm.tolist(),
            "feature_importance": {
                feature_names[i]: importances[i] for i in range(len(feature_names))
            }
        }

        json_path = f"{base_path}_report.json"
        import json
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        return json_path

    except Exception as e:
        logger.error(f"Failed to save JSON report: {str(e)}", exc_info=True)
        return ""