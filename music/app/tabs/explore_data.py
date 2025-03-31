import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def render_explore_tab():
    """Render the Data Exploration tab"""
    st.subheader("Music Preference Data Exploration")

    data = st.session_state.get('current_data')

    if data is None:
        st.info("Please upload a CSV file to explore data.")
        return

    # Make a copy to avoid modifying the original
    df = data.copy()

    # Determine the genre column name
    genre_column = 'genre' if 'genre' in df.columns else 'music_preference'

    # Add filter controls within the tab (not in sidebar)
    st.subheader("Data Filters")

    # Create a container for filters
    filter_container = st.container()

    with filter_container:
        col1, col2 = st.columns(2)

        with col1:
            # Age filter
            min_age = int(df['age'].min())
            max_age = int(df['age'].max())
            age_range = st.slider(
                "Age Range",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age)
            )

        with col2:
            # Gender filter
            all_genders = sorted(df['gender'].unique())
            selected_genders = st.multiselect(
                "Gender",
                options=all_genders,
                default=all_genders,
                format_func=lambda x: "Male" if x == 1 else "Female"
            )

        # Genre filter
        all_genres = sorted(df[genre_column].unique())
        selected_genres = st.multiselect(
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
        render_genre_distribution(filtered_df, genre_column)

    with viz_tab2:
        render_age_distribution(filtered_df)

    with viz_tab3:
        render_combined_analysis(filtered_df, genre_column)


def render_genre_distribution(df, genre_column):
    """
    Render genre distribution visualizations

    Args:
        df (pd.DataFrame): Filtered DataFrame
        genre_column (str): Name of the genre column
    """
    st.subheader("Genre Distribution")

    if df.empty:
        st.warning("No data available with current filters.")
        return

    # Genre counts
    genre_counts = df[genre_column].value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']

    # Display as bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Genre', y='Count', data=genre_counts, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # Display as table
    st.dataframe(genre_counts, use_container_width=True)


def render_age_distribution(df):
    """
    Render age distribution visualizations

    Args:
        df (pd.DataFrame): Filtered DataFrame
    """
    st.subheader("Age Distribution")

    if df.empty:
        st.warning("No data available with current filters.")
        return

    # Create age groups for better visualization
    filtered_df = df.copy()
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


def render_combined_analysis(df, genre_column):
    """
    Render combined analysis visualizations

    Args:
        df (pd.DataFrame): Filtered DataFrame
        genre_column (str): Name of the genre column
    """
    st.subheader("Combined Analysis")

    if df.empty:
        st.warning("No data available with current filters.")
        return

    # Create a copy for manipulation
    filtered_df = df.copy()

    # Create pivot table of genre by gender and age group
    filtered_df['gender_label'] = filtered_df['gender'].map({0: 'Female', 1: 'Male'})
    filtered_df['age_group'] = pd.cut(
        filtered_df['age'],
        bins=st.session_state.config['data']['age_bins'],
        labels=st.session_state.config['data']['age_labels'],
        right=False
    )

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

    if not genre_gender_counts.empty:
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