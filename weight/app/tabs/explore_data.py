import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def render_explore_tab():
    """Render the Data Exploration tab"""
    st.subheader("Weight Data Exploration")

    data = st.session_state.get('current_data')

    if data is None:
        st.info("Please upload a CSV file to explore data.")
        return

    # Make a copy to avoid modifying the original
    df = data.copy()

    # Determine the weight column name
    weight_column = 'Weight' if 'Weight' in df.columns else 'weight_predictor'

    # Add filter controls within the tab
    st.subheader("Data Filters")

    filter_container = st.container()

    with filter_container:
        col1, col2, col3 = st.columns(3)

        with col2:
            min_height = int(df['Height'].min())
            max_height = int(df['Height'].max())
            height_range = st.slider(
                "Height Range",
                min_value=min_height,
                max_value=max_height,
                value=(min_height, max_height)
            )

        with col1:
            all_genders = sorted(df['Gender'].unique())
            selected_genders = st.multiselect(
                "Gender",
                options=all_genders,
                default=all_genders,
                format_func=lambda x: "Male" if x == 1 else "Female"
            )
        with col2:
            min_weight = int(df['Weight'].min())
            max_weight = int(df['Weight'].max())
            weight_range = st.slider(
                "Weight Range",
                min_value=min_weight,
                max_value=max_weight,
                value=(min_weight, max_weight)
            )

   

    # Apply filters
    filtered_df = df[
        (df['Height'] >= height_range[0]) &
        (df['Height'] <= height_range[1]) &
        (df['Gender'].isin(selected_genders)) &
         (df['Weight'] >= weight_range[0]) &
        (df['Weight'] <= weight_range[1]) 
    ]

    st.write(f"Filtered data: {len(filtered_df)} records")

    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Weight Distribution", "Height Distribution", "Combined Analysis"])

    with viz_tab1:
        render_weight_distribution(filtered_df, weight_column)

    with viz_tab2:
        render_height_distribution(filtered_df)

    with viz_tab3:
        render_combined_analysis(filtered_df, weight_column)


def render_weight_distribution(df, weight_column):
    """
    Render weight distribution visualizations

    Args:
        df (pd.DataFrame): Filtered DataFrame
        weight_column (str): Name of the weight column
    """
    st.subheader("Weight Distribution")

    if df.empty:
        st.warning("No data available with current filters.")
        return

    weight_counts = df[weight_column].value_counts().reset_index()
    weight_counts.columns = ['Weight', 'Count']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Weight', y='Count', data=weight_counts, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(weight_counts, use_container_width=True)


def render_height_distribution(df):
    """
    Render height distribution visualizations

    Args:
        df (pd.DataFrame): Filtered DataFrame
    """
    st.subheader("Height Distribution")

    if df.empty:
        st.warning("No data available with current filters.")
        return

    filtered_df = df.copy()
    filtered_df['height_group'] = pd.cut(
        filtered_df['Height'],
        bins=st.session_state.config['data']['height_bins'],
        labels=st.session_state.config['data']['height_labels'],
        right=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=filtered_df,
        x='Height',
        hue='Gender',
        multiple='stack',
        palette=['pink', 'blue'],
        bins=20,
        ax=ax
    )
    plt.xlabel('Height')
    plt.ylabel('Count')
    plt.title('Height Distribution by Gender')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Female', 'Male'])
    st.pyplot(fig)

    height_group_counts = filtered_df['height_group'].value_counts().reset_index()
    height_group_counts.columns = ['Height Group', 'Count']

    st.dataframe(height_group_counts, use_container_width=True)


def render_combined_analysis(df, weight_column):
    """
    Render combined analysis visualizations

    Args:
        df (pd.DataFrame): Filtered DataFrame
        weight_column (str): Name of the weight column
    """
    st.subheader("Combined Analysis")

    if df.empty:
        st.warning("No data available with current filters.")
        return

    filtered_df = df.copy()
    filtered_df['gender_label'] = filtered_df['Gender'].map({0: 'Female', 1: 'Male'})
    filtered_df['height_group'] = pd.cut(
        filtered_df['Height'],
        bins=st.session_state.config['data']['height_bins'],
        labels=st.session_state.config['data']['height_labels'],
        right=False
    )

    pivot_df = pd.crosstab(
        [filtered_df['gender_label'], filtered_df['height_group']],
        filtered_df[weight_column]
    )

    if not pivot_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='d', linewidths=.5, ax=ax)
        plt.title('Weight Distribution by Gender and Height')
        plt.tight_layout()
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    weight_gender_counts = filtered_df.groupby(['gender_label', weight_column]).size().unstack()

    if not weight_gender_counts.empty:
        weight_gender_counts.plot(kind='bar', ax=ax)
        plt.title('Weight Distribution by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title='Weight')
        plt.tight_layout()
        st.pyplot(fig)

    if st.checkbox("Show Raw Data"):
        st.write(filtered_df)
