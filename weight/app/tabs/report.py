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
            timestamp_match = re.search(r'model_(\d+_\d+)', selected_model)
            if timestamp_match:
                model_timestamp = timestamp_match.group(1)
                matching_reports = [f for f in metric_files if model_timestamp in f]
                if matching_reports:
                    latest_metrics = matching_reports[0]
                else:
                    st.warning(f"No specific report found for model {selected_model}. Showing most recent report.")
                    latest_metrics = sorted(metric_files)[-1]
            else:
                latest_metrics = sorted(metric_files)[-1]
        else:
            latest_metrics = sorted(metric_files)[-1]

        latest_timestamp = latest_metrics.replace('_metrics.txt', '')

        # Check for report files: scatter, residuals, coefficients, gender plot, json, pdf
        scatter_files = [f for f in os.listdir(reports_dir) if f.endswith('_scatter.png')]
        gender_plot_files = [f for f in os.listdir(reports_dir) if f.endswith('_gender_height_weight.png')]
        residual_files = [f for f in os.listdir(reports_dir) if f.endswith('_residuals.png')]
        coef_files = [f for f in os.listdir(reports_dir) if f.endswith('_coefficients.png')]
        json_files = [f for f in os.listdir(reports_dir) if f.endswith('_report.json')]
        pdf_files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf')]

        # Display metrics in tables
        st.subheader("Regression Model Training Report")

        if selected_model is None and st.session_state.get('using_newly_trained_model', False):
            st.success("Viewing report for newly trained model")
        elif selected_model:
            st.info(f"Showing training report for model: {selected_model}")

        # Try to load JSON data if available
        json_data_loaded = False
        matching_json = [f for f in json_file if latest_timestamp in f]

        if matching_json:
            json_path = os.path.join(reports_dir, matching_json[0])
            try:
                with open(json_path, 'r') as f:
                    report_data = json.load(f)
                json_data_loaded = True

                # Add PDF download button if available
                matching_pdf = [f for f in pdf_file if latest_timestamp in f]
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
                    "Metric": [
                        "Training Date", "Model Type",
                        "Training Samples", "Testing Samples",
                        "Training MSE", "Training MAE", "Training R²",
                        "Testing MSE", "Testing MAE", "Testing R²"
                    ],
                    "Value": [
                        report_data.get("timestamp", "N/A"),
                        report_data.get("model_type", "N/A"),
                        report_data.get("training_samples", "N/A"),
                        report_data.get("testing_samples", "N/A"),
                        f"{report_data.get('train_metrics', {}).get('mse', 0):.4f}",
                        f"{report_data.get('train_metrics', {}).get('mae', 0):.4f}",
                        f"{report_data.get('train_metrics', {}).get('r2', 0):.4f}",
                        f"{report_data.get('test_metrics', {}).get('mse', 0):.4f}",
                        f"{report_data.get('test_metrics', {}).get('mae', 0):.4f}",
                        f"{report_data.get('test_metrics', {}).get('r2', 0):.4f}"
                    ]
                }
                st.table(pd.DataFrame(info_data))
            except Exception as e:
                st.warning(f"Could not load detailed metrics data from JSON: {str(e)}")
                json_data_loaded = False

        if not json_data_loaded:
            st.info("Basic training report available. Detailed metrics might be missing.")
            metrics_path = os.path.join(reports_dir, latest_metrics)
            try:
                with open(metrics_path, 'r') as f:
                    metrics_content = f.read()
                st.text(metrics_content)
            except Exception as e:
                st.warning(f"Could not load metrics file content: {str(e)}")

        # Display visualizations
        display_report_visualizations(reports_dir, scatter_files, gender_plot_files, residual_files, coef_files,
                                      latest_timestamp)

    except Exception as e:
        st.error(f"Error displaying training report: {str(e)}")


def display_report_visualizations(reports_dir, scatter_files, gender_plot_files, residual_files, coef_files,
                                  latest_timestamp):

    """
    Display report visualizations

    Args:
        reports_dir (str): Directory containing report files
        scatter_files (list): List of scatter plot files
        gender_plot_files (list): List of gender height-weight plot files
        residual_files (list): List of residual plot files
        coef_files (list): List of coefficient plot files
        latest_timestamp (str): Timestamp for the model's report
    """
    # Display scatter plot if available
    matching_comparison = [f for f in comparison_file if latest_timestamp in f]
    if matching_comparison:
        try:
            comparison_path = os.path.join(reports_dir, matching_comparison[0])
            st.subheader("Actual vs. Predicted Scatter Plot")
            scatter_img = mpimg.imread(comparison_path)
            st.image(scatter_img, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load scatter plot visualization: {str(e)}")

    matching_height_scatter = [f for f in height_scatter_file if latest_timestamp in f]
    if matching_height_scatter:
        try:
            height_scatter_path = os.path.join(reports_dir, matching_height_scatter[0])
            st.subheader("Height vs Weight with Line of Best Fitt")
            scatter_img = mpimg.imread(height_scatter_path)
            st.image(scatter_img, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load scatter plot visualization: {str(e)}")

    matching_gender_scatter = [f for f in  gender_scatter_file if latest_timestamp in f]
    if matching_gender_scatter:
        try:
            gender_scatter_path = os.path.join(reports_dir, matching_gender_scatter[0])
            st.subheader("Gender vs Weight")
            scatter_img = mpimg.imread(gender_scatter_path)
            st.image(scatter_img, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load scatter plot visualization: {str(e)}")

    # Display Weight vs Height by Gender plot if available
    matching_gender_plot = [f for f in gender_plot_files if latest_timestamp in f]
    if matching_gender_plot:
        try:
            gender_plot_path = os.path.join(reports_dir, matching_gender_plot[0])
            st.subheader("Weight vs Height by Gender")
            gender_plot_img = mpimg.imread(gender_plot_path)
            st.image(gender_plot_img, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load weight vs height by gender plot: {str(e)}")

    # Display residual plot if available
    matching_residual = [f for f in residual_file if latest_timestamp in f]
    if matching_residual:
        try:
            residual_path = os.path.join(reports_dir, matching_residual[0])
            st.subheader("Residual Plot")
            residual_img = mpimg.imread(residual_path)
            st.image(residual_img, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load residual plot visualization: {str(e)}")

    # Display coefficient plot if available
    matching_coef = [f for f in coef_file if latest_timestamp in f]
    if matching_coef:
        try:
            coef_path = os.path.join(reports_dir, matching_coef[0])
            st.subheader("Feature Coefficient Plot")
            coef_img = mpimg.imread(coef_path)
            st.image(coef_img, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load coefficient plot visualization: {str(e)}")
