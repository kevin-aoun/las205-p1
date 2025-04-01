import streamlit as st
import os
import json
import re
import matplotlib.image as mpimg
import pandas as pd

def render_report_tab(model_exists):
    """
    Render the Training Report tab

    Args:
        model_exists (bool): Whether a usable model exists
    """
    # Check if we've just trained a new model
    using_newly_trained = st.session_state.get('using_newly_trained_model', False)

    if st.session_state.get('show_training_report', False) or model_exists:
        if using_newly_trained:
            display_training_report()
        else:
            selected_model = st.session_state.get('selected_model')
            display_training_report(selected_model)
    else:
        st.info("No training reports available. Train a model first to see the training report.")


def display_training_report(selected_model=None):
    """
    Display the training report for the selected model

    Args:
        selected_model (str, optional): The selected model filename
    """
    config = st.session_state.config
    models_dir = config['model']['models_dir']
    reports_dir = os.path.join(models_dir, 'train_reports')

    if not os.path.exists(reports_dir):
        st.info("No training reports available.")
        return

    # Try to find report files
    metric_files = [f for f in os.listdir(reports_dir) if f.endswith('_metrics.txt')]

    if not metric_files:
        st.info("No training reports available.")
        return

    try:
        # If a specific model is selected, find its corresponding report
        if selected_model:
            # Extract timestamp from the model filename
            # Typically model names look like 'music_preferences_model_20250401_123456.joblib'
            # We want to extract the '20250401_123456' part
            timestamp_match = re.search(r'model_(\d+_\d+)', selected_model)
            if timestamp_match:
                model_timestamp = timestamp_match.group(1)
                # Find reports matching this timestamp
                matching_reports = [f for f in metric_files if model_timestamp in f]
                if matching_reports:
                    latest_metrics = matching_reports[0]
                else:
                    # Fall back to most recent if no matching report
                    st.warning(f"No specific report found for model {selected_model}. Showing most recent report.")
                    latest_metrics = sorted(metric_files)[-1]
            else:
                # Fall back to most recent if pattern doesn't match
                latest_metrics = sorted(metric_files)[-1]
        else:
            # No model specified, use most recent report
            # Sort by name to ensure most recent timestamp is last
            latest_metrics = sorted(metric_files)[-1]

        latest_timestamp = latest_metrics.replace('_metrics.txt', '')

        # Check for report files
        cm_files = [f for f in os.listdir(reports_dir) if f.endswith('_confusion_matrix.png')]
        fi_files = [f for f in os.listdir(reports_dir) if f.endswith('_feature_importance.png')]
        json_files = [f for f in os.listdir(reports_dir) if f.endswith('_report.json')]
        tree_files = [f for f in os.listdir(reports_dir) if f.endswith('_tree_viz.png')]
        pdf_files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf')]

        # Display metrics in tables
        st.subheader("Model Training Report")

        # Show appropriate message based on model source
        if selected_model is None and st.session_state.get('using_newly_trained_model', False):
            st.success("Viewing report for newly trained model")
        elif selected_model:
            st.info(f"Showing training report for model: {selected_model}")

        # Try to load JSON data if available
        json_data_loaded = False
        matching_json = [f for f in json_files if latest_timestamp in f]

        if matching_json:
            json_path = os.path.join(reports_dir, matching_json[0])

            try:
                with open(json_path, 'r') as f:
                    report_data = json.load(f)
                json_data_loaded = True

                # Add PDF download button if available
                matching_pdf = [f for f in pdf_files if latest_timestamp in f]
                if matching_pdf:
                    pdf_path = os.path.join(reports_dir, matching_pdf[0])
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
                    "Metric": ["Training Date", "Accuracy", "F1 Score (weighted)", "Precision (weighted)",
                               "Recall (weighted)"],
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

                if class_data:
                    st.table(pd.DataFrame(class_data))
            except Exception as e:
                st.warning(f"Could not load detailed metrics data from JSON: {str(e)}")
                json_data_loaded = False

        # If JSON data couldn't be loaded, display a basic message
        if not json_data_loaded:
            st.info("Basic training report available. Some detailed metrics might be missing.")
            metrics_path = os.path.join(reports_dir, latest_metrics)

            try:
                with open(metrics_path, 'r') as f:
                    metrics_content = f.read()
                st.text(metrics_content)
            except Exception as e:
                st.warning(f"Could not load metrics file content: {str(e)}")

        # Display visualizations
        display_report_visualizations(reports_dir, cm_files, fi_files, tree_files, latest_timestamp)

    except Exception as e:
        st.error(f"Error displaying training report: {str(e)}")


def display_report_visualizations(reports_dir, cm_files, fi_files, tree_files, latest_timestamp):
    """
    Display report visualizations

    Args:
        reports_dir (str): Directory containing report files
        cm_files (list): List of confusion matrix files
        fi_files (list): List of feature importance files
        tree_files (list): List of tree visualization files
        latest_timestamp (str): Timestamp for the model's report
    """
    # Display confusion matrix if available
    matching_cm = [f for f in cm_files if latest_timestamp in f]
    if matching_cm:
        try:
            cm_path = os.path.join(reports_dir, matching_cm[0])
            st.subheader("Confusion Matrix")
            cm_img = mpimg.imread(cm_path)
            st.image(cm_img, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load confusion matrix visualization: {str(e)}")

    # Display feature importance if available
    matching_fi = [f for f in fi_files if latest_timestamp in f]
    if matching_fi:
        try:
            fi_path = os.path.join(reports_dir, matching_fi[0])
            st.subheader("Feature Importance")
            fi_img = mpimg.imread(fi_path)
            st.image(fi_img, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load feature importance visualization: {str(e)}")

    # Display tree visualization if available
    matching_tree = [f for f in tree_files if latest_timestamp in f]
    if matching_tree:
        try:
            tree_path = os.path.join(reports_dir, matching_tree[0])
            st.subheader("Decision Tree Visualization")
            st.write("Sample of one tree from the Random Forest:")
            tree_img = mpimg.imread(tree_path)
            st.image(tree_img, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load tree visualization: {str(e)}")