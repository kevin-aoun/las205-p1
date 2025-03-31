import streamlit as st
import os
import json
import matplotlib.image as mpimg
import pandas as pd


def render_report_tab(model_exists):
    """
    Render the Training Report tab

    Args:
        model_exists (bool): Whether a usable model exists
    """
    if st.session_state.get('show_training_report', False) or model_exists:
        display_training_report()
    else:
        st.info("No training reports available. Train a model first to see the training report.")


def display_training_report():
    """
    Display the most recent training report if it exists.
    """
    config = st.session_state.config
    models_dir = config['model']['models_dir']
    reports_dir = os.path.join(models_dir, 'train_reports')

    if not os.path.exists(reports_dir):
        st.info("No training reports available.")
        return

    # Find the most recent report files
    metric_files = [f for f in os.listdir(reports_dir) if f.endswith('_metrics.txt')]
    cm_files = [f for f in os.listdir(reports_dir) if f.endswith('_confusion_matrix.png')]
    fi_files = [f for f in os.listdir(reports_dir) if f.endswith('_feature_importance.png')]
    json_files = [f for f in os.listdir(reports_dir) if f.endswith('_report.json')]
    tree_files = [f for f in os.listdir(reports_dir) if f.startswith('tree_viz_') and f.endswith('.png')]
    pdf_files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf')]

    if not metric_files or not json_files:
        st.info("No training reports available.")
        return

    # Get the most recent files
    latest_metrics = sorted(metric_files)[-1]
    latest_json = sorted(json_files)[-1]
    latest_timestamp = latest_metrics.replace('_metrics.txt', '')

    # Load JSON data for metrics table
    with open(os.path.join(reports_dir, latest_json), 'r') as f:
        report_data = json.load(f)

    # Display metrics in tables
    st.subheader("Model Training Report")

    # Add PDF download button
    latest_pdf = [f for f in pdf_files if latest_timestamp.replace('_metrics.txt', '') in f]
    if latest_pdf:
        pdf_path = os.path.join(reports_dir, latest_pdf[0])
        with open(pdf_path, "rb") as file:
            pdf_data = file.read()

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_data,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf"
        )

    # Display basic model information
    st.write("### Model Information")
    info_data = {
        "Metric": ["Training Date", "Accuracy", "F1 Score (weighted)", "Precision (weighted)", "Recall (weighted)"],
        "Value": [
            report_data["timestamp"],
            f"{report_data['accuracy']:.4f}",
            f"{report_data['f1']['weighted']:.4f}",
            f"{report_data['precision']['weighted']:.4f}",
            f"{report_data['recall']['weighted']:.4f}"
        ]
    }
    st.table(pd.DataFrame(info_data))

    # Display per-class metrics
    st.write("### Per-Class Performance")

    # Create a DataFrame for class metrics
    class_data = []
    for class_name, metrics in report_data["class_report"].items():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        class_data.append({
            "Class": class_name,
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1 Score": f"{metrics['f1-score']:.4f}",
            "Support": metrics['support']
        })

    st.table(pd.DataFrame(class_data))

    # Display visualizations
    display_report_visualizations(reports_dir, cm_files, fi_files, tree_files, latest_timestamp)

    # Add a link to the full metrics text file
    metrics_path = os.path.join(reports_dir, latest_metrics)
    with open(metrics_path, 'r') as f:
        metrics_content = f.read()

    with st.expander("View Full Metrics Report"):
        st.text(metrics_content)


def display_report_visualizations(reports_dir, cm_files, fi_files, tree_files, latest_timestamp):
    """
    Display report visualizations

    Args:
        reports_dir (str): Directory containing report files
        cm_files (list): List of confusion matrix files
        fi_files (list): List of feature importance files
        tree_files (list): List of tree visualization files
        latest_timestamp (str): Timestamp for the most recent report
    """
    # Display confusion matrix if available
    cm_file = [f for f in cm_files if latest_timestamp in f]
    if cm_file:
        cm_path = os.path.join(reports_dir, cm_file[0])
        st.subheader("Confusion Matrix")
        cm_img = mpimg.imread(cm_path)
        st.image(cm_img, use_column_width=True)

    # Display feature importance if available
    fi_file = [f for f in fi_files if latest_timestamp in f]
    if fi_file:
        fi_path = os.path.join(reports_dir, fi_file[0])
        st.subheader("Feature Importance")
        fi_img = mpimg.imread(fi_path)
        st.image(fi_img, use_column_width=True)

    # Display tree visualization if available
    if tree_files:
        latest_tree = sorted(tree_files)[-1]
        tree_path = os.path.join(reports_dir, latest_tree)
        st.subheader("Decision Tree Visualization")
        st.write("Sample of one tree from the Random Forest:")
        tree_img = mpimg.imread(tree_path)
        st.image(tree_img, use_column_width=True)