# Music Preference Predictor

A Streamlit application that predicts music genre preferences based on age and gender using both statistical and machine learning approaches.

## Features

- CSV file upload for training data
- User input for age and gender
- Two prediction methods:
  - Percentage-based statistical model
  - Machine learning model using Random Forest

## File Structure

- `app.py`: Main Streamlit application with UI components
- `prediction.py`: Contains prediction functions using percentage and ML approaches
- `requirements.txt`: List of dependencies
- `models/`: Directory where trained ML models and label encoders are saved

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Upload a CSV file with columns: age, gender, genre
3. Enter your age and gender
4. Click "Predict Music Preference" to see the results

## CSV Format

The CSV file should contain the following columns:
- `age`: Numeric age of the person
- `gender`: Binary gender (1 for Male, 0 for Female)
- `genre`: Music genre preference (e.g., HipHop, Jazz, Classical, Dance, Acoustic)

Example:
```
age,gender,genre
20,1,HipHop
23,1,HipHop
25,1,HipHop
26,1,Jazz
...
```

## How It Works

### Age Binning
When a CSV file is uploaded, ages are automatically binned into three groups:
- Young: Ages ≤ 25
- Mid: Ages 26-31
- Older: Ages > 31

### Percentage-based Prediction
This method groups users by age bins and gender, then calculates the most common genre for each group. If a user's age is outside the range in the dataset, it assigns them to the closest age group.

### Machine Learning Prediction
This method uses a Random Forest classifier trained on the uploaded data to predict genre preferences. The model is:
- Trained only once when the CSV is uploaded
- Saved to disk in the 'models' directory with a timestamp
- Automatically loaded from disk in future sessions
- Reused for all subsequent predictions