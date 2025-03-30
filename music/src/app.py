import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import base64

from utils import (
    logger, init_config,
    save_uploaded_file, check_saved_model_exists, config_sidebar
)
from prediction import predict_using_percentage, predict_using_ml, train_and_save_model


def display_percentage_results(percentage_result):
    """
    Display all the percentages from the percentage-based prediction.

    Args:
        percentage_result (dict): Result from predict_using_percentage
    """
    st.write(f"**Predicted Genre:** {percentage_result['genre']}")
    st.write(f"**Top Confidence:** {percentage_result['confidence']:.2f}%")

    # If we have all percentages, display them as a table
    if 'all_percentages' in percentage_result and percentage_result['all_percentages']:
        st.write("**All Genre Percentages:**")

        # Get the percentages
        genres = list(percentage_result['all_percentages'].keys())
        percentages = list(percentage_result['all_percentages'].values())

        # Sort by percentage (descending)
        sorted_indices = sorted(range(len(percentages)), key=lambda i: percentages[i], reverse=True)
        sorted_genres = [genres[i] for i in sorted_indices]
        sorted_percentages = [percentages[i] for i in sorted_indices]

        # Create and display the table
        percentage_df = pd.DataFrame({
            'Genre': sorted_genres,
            'Percentage': [f"{p:.2f}%" for p in sorted_percentages]
        })
        st.dataframe(percentage_df, use_container_width=True)


def create_download_link(file_path, link_text):
    """
    Create a download link for a file

    Args:
        file_path (str): Path to the file to be downloaded
        link_text (str): Text to display on the download link

    Returns:
        str: HTML link for downloading the file
    """
    with open(file_path, "rb") as f:
        data = f.read()

    b64 = base64.b64encode(data).decode()
    file_name = os.path.basename(file_path)

    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{link_text}</a>'
    return href


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
    import json
    with open(os.path.join(reports_dir, latest_json), 'r') as f:
        report_data = json.load(f)

    # Display metrics in tables
    st.subheader("Model Training Report")

    # Add PDF download link if available
    latest_pdf = [f for f in pdf_files if latest_timestamp.replace('_metrics.txt', '') in f]
    if latest_pdf:
        pdf_path = os.path.join(reports_dir, latest_pdf[0])
        st.markdown(create_download_link(pdf_path, "📥 Download PDF Report"), unsafe_allow_html=True)

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

    # Add a link to the full metrics text file
    metrics_path = os.path.join(reports_dir, latest_metrics)
    with open(metrics_path, 'r') as f:
        metrics_content = f.read()

    with st.expander("View Full Metrics Report"):
        st.text(metrics_content)


def explore_data_tab(data):
    """
    Display data exploration visualizations and filters.

    Args:
        data (pd.DataFrame): Data to explore
    """
    st.subheader("Music Preference Data Exploration")

    if data is None:
        st.info("Please upload a CSV file to explore data.")
        return

    # Make a copy to avoid modifying the original
    df = data.copy()

    # Determine the genre column name
    genre_column = 'genre' if 'genre' in df.columns else 'music_preference'

    # Add filter sidebar
    st.sidebar.subheader("Data Filters")

    # Age filter
    min_age = int(df['age'].min())
    max_age = int(df['age'].max())
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

    # Gender filter
    all_genders = sorted(df['gender'].unique())
    selected_genders = st.sidebar.multiselect(
        "Gender",
        options=all_genders,
        default=all_genders,
        format_func=lambda x: "Male" if x == 1 else "Female"
    )

    # Genre filter
    all_genres = sorted(df[genre_column].unique())
    selected_genres = st.sidebar.multiselect(
        "Music Genre",
        options=all_genres,
        default=all_genres
    )

    # Apply filters
    filtered_df = df[
        (df['age'] >= age_range[0]) &
        (df['age'] <= age_range[1]) &
        (df['gender'].isin(selected_genders)) &
        (df[genre_column].isin(selected_genres))
        ]

    # Display filtered data summary
    st.write(f"Filtered data: {len(filtered_df)} records")

    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Genre Distribution", "Age Distribution", "Combined Analysis"])

    with viz_tab1:
        st.subheader("Genre Distribution")

        # Genre counts
        genre_counts = filtered_df[genre_column].value_counts().reset_index()
        genre_counts.columns = ['Genre', 'Count']

        # Display as bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Genre', y='Count', data=genre_counts, ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        # Display as table
        st.dataframe(genre_counts, use_container_width=True)

    with viz_tab2:
        st.subheader("Age Distribution")

        # Create age groups for better visualization
        filtered_df['age_group'] = pd.cut(
            filtered_df['age'],
            bins=st.session_state.config['data']['age_bins'],
            labels=st.session_state.config['data']['age_labels'],
            right=False
        )

        # Age distribution by gender
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=filtered_df,
            x='age',
            hue='gender',
            multiple='stack',
            palette=['pink', 'blue'],
            bins=20,
            ax=ax
        )
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.title('Age Distribution by Gender')
        # Add legend with custom labels
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['Female', 'Male'])
        st.pyplot(fig)

        # Age group counts
        age_group_counts = filtered_df['age_group'].value_counts().reset_index()
        age_group_counts.columns = ['Age Group', 'Count']

        # Display as table
        st.dataframe(age_group_counts, use_container_width=True)

    with viz_tab3:
        st.subheader("Combined Analysis")

        # Create pivot table of genre by gender and age group
        filtered_df['gender_label'] = filtered_df['gender'].map({0: 'Female', 1: 'Male'})

        pivot_df = pd.crosstab(
            [filtered_df['gender_label'], filtered_df['age_group']],
            filtered_df[genre_column]
        )

        # Plot heatmap
        if not pivot_df.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='d', linewidths=.5, ax=ax)
            plt.title('Genre Distribution by Gender and Age Group')
            plt.tight_layout()
            st.pyplot(fig)

        # Bar chart showing genre distribution by gender
        fig, ax = plt.subplots(figsize=(12, 6))
        genre_gender_counts = filtered_df.groupby(['gender_label', genre_column]).size().unstack()
        genre_gender_counts.plot(kind='bar', ax=ax)
        plt.title('Genre Distribution by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title='Genre')
        plt.tight_layout()
        st.pyplot(fig)

        # Display the raw data with filters applied
        if st.checkbox("Show Raw Data"):
            st.write(filtered_df)


def main():
    # Initialize configuration in session state
    init_config()
    config = st.session_state.config

    # Initialize session state variables
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False
    if 'file_processed' not in st.session_state:
        st.session_state['file_processed'] = False
    if 'current_data' not in st.session_state:
        st.session_state['current_data'] = None
    if 'trained_model' not in st.session_state:
        st.session_state['trained_model'] = None
    if 'label_encoder' not in st.session_state:
        st.session_state['label_encoder'] = None
    if 'show_training_report' not in st.session_state:
        st.session_state['show_training_report'] = False

    # Set page title
    st.title(config['app']['title'])

    # Handle configuration sidebar
    config_updated = config_sidebar()

    # Reload app if configuration was updated
    if config_updated:
        st.rerun()

    # Check if a saved model exists
    latest_model = check_saved_model_exists()
    model_exists = latest_model is not None and config['model']['use_saved_model']

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data & Model", "Prediction", "Training Report", "Explore Data"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Upload")
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='file_uploader')

            if uploaded_file is None:
                st.error("CSV upload is mandatory. Please upload a file to continue.")

        with col2:
            st.subheader("Model Status")
            if model_exists:
                st.success(f"Using saved model: {latest_model}")
                if st.button("Load saved model"):
                    st.session_state['model_trained'] = True
                    st.success("Saved model loaded successfully")

        # Process uploaded file if it exists
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                logger.info(f"File uploaded: {uploaded_file.name}, {df.shape[0]} rows, {df.shape[1]} columns")

                # Save the file if enabled in config
                if config['data']['save_uploads']:
                    save_path = save_uploaded_file(uploaded_file)
                    if save_path:
                        st.success(f"File saved successfully")

                # Store data in session state
                st.session_state['current_data'] = df

                if 'age' in df.columns:
                    st.session_state['min_age'] = df['age'].min()
                    st.session_state['max_age'] = df['age'].max()

                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())

                # Handle model training decision
                if model_exists and not st.session_state.get('file_processed', False):
                    st.subheader("Model Training Options")
                    st.info("A saved model exists. You can use it or train a new model with the uploaded data.")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Use existing model"):
                            st.session_state['file_processed'] = True
                            st.session_state['model_trained'] = True
                            st.success("Using existing model with new data")
                    with col2:
                        if st.button("Train new model"):
                            with st.spinner("Training new model..."):
                                model, le = train_and_save_model(df, save_model=config['model']['save_model'])
                                st.session_state['trained_model'] = model
                                st.session_state['label_encoder'] = le
                                st.session_state['model_trained'] = True
                                st.session_state['file_processed'] = True
                                st.session_state['show_training_report'] = True

                            if config['model']['save_model']:
                                st.success("New model trained and saved successfully")
                            else:
                                st.success("New model trained successfully (not saved to disk)")

                            st.info("View the training report in the 'Training Report' tab")

                # Automatically train model if no existing model
                elif not model_exists and not st.session_state.get('file_processed', False):
                    st.info("No existing model found. Training a new model...")
                    with st.spinner("Training new model..."):
                        model, le = train_and_save_model(df, save_model=config['model']['save_model'])
                        st.session_state['trained_model'] = model
                        st.session_state['label_encoder'] = le
                        st.session_state['model_trained'] = True
                        st.session_state['file_processed'] = True
                        st.session_state['show_training_report'] = True

                    if config['model']['save_model']:
                        st.success("Model trained and saved successfully")
                    else:
                        st.success("Model trained successfully (not saved to disk)")

                    st.info("View the training report in the 'Training Report' tab")

            except Exception as e:
                logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
                st.error(f"Error processing file: {str(e)}")
        else:
            # Reset processing flags when no file is uploaded
            st.session_state['file_processed'] = False

    with tab2:
        # Only show prediction UI if a file has been uploaded
        if st.session_state.get('current_data') is not None:
            st.subheader("Enter Your Information")

            gender = st.radio("Select your gender:",
                              options=["Male", "Female"],
                              index=0)
            gender_value = 1 if gender == "Male" else 0

            age_min, age_max = config['app']['age_range']
            default_age = config['app']['default_age']
            age = st.slider("Select your age:", age_min, age_max, default_age)

            predict_button = st.button("Predict Music Preference")

            if predict_button:
                can_predict = True

                # Check if we can use ML model
                can_use_ml = st.session_state.get('model_trained', False) or model_exists

                # We always have current data at this point
                can_use_percentage = True
                data = st.session_state['current_data']

                if not can_use_ml and not can_use_percentage:
                    st.warning("No model or data available for prediction.")
                    can_predict = False

                if can_predict:
                    st.subheader("Prediction Results")

                    col1, col2 = st.columns(2)

                    # Percentage-based prediction
                    with col1:
                        st.info("Based on Percentage Model")
                        percentage_result = predict_using_percentage(data, age, gender_value)
                        display_percentage_results(percentage_result)

                    # ML-based prediction
                    with col2:
                        st.info("Based on Machine Learning Model")
                        if can_use_ml:
                            if st.session_state.get('trained_model') is not None and st.session_state.get(
                                    'label_encoder') is not None:
                                # Use the model from session state
                                ml_result = predict_using_ml(
                                    None,
                                    age,
                                    gender_value,
                                    st.session_state['trained_model'],
                                    st.session_state['label_encoder']
                                )
                            else:
                                # Load the saved model
                                ml_result = predict_using_ml(data, age, gender_value)

                            st.write(f"**Predicted Genre:** {ml_result['genre']}")
                            st.write(f"**Confidence:** {ml_result['confidence']:.2f}%")

                            # Display probabilities if available
                            if 'probabilities' in ml_result and ml_result['probabilities']:
                                # Create a bar chart for probabilities
                                fig, ax = plt.subplots(figsize=(10, 5))
                                genres = list(ml_result['probabilities'].keys())
                                probs = [ml_result['probabilities'][g] * 100 for g in genres]

                                # Sort by probability (descending)
                                sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
                                sorted_genres = [genres[i] for i in sorted_indices]
                                sorted_probs = [probs[i] for i in sorted_indices]

                                # Plot bars
                                bars = ax.bar(sorted_genres, sorted_probs, color='lightgreen')

                                # Add percentage labels on top of bars
                                for bar in bars:
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                                            f'{height:.1f}%', ha='center', va='bottom')

                                ax.set_ylim(0, max(probs) * 1.1)
                                ax.set_ylabel('Probability (%)')
                                ax.set_title('Genre Probabilities from ML Model')
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()

                                st.pyplot(fig)
                        else:
                            st.write("No ML model available for prediction")

                    logger.info(f"Predictions displayed for age={age}, gender={gender_value}")
        else:
            st.info("Please upload a CSV file to make predictions.")

    with tab3:
        if st.session_state.get('show_training_report', False) or model_exists:
            display_training_report()
        else:
            st.info("No training reports available. Train a model first to see the training report.")

    with tab4:
        explore_data_tab(st.session_state.get('current_data'))


if __name__ == "__main__":
    main()