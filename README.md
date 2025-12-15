# Airline Performance Dashboard

An interactive Streamlit dashboard for analyzing US domestic flight delays with machine learning predictions.

## Features

- **Delay Analysis**: Comprehensive breakdown of delay types and causes
- **Time Analysis**: Temporal patterns in flight volumes and delays
- **Airline Analysis**: Performance metrics by carrier
- **Airport Analysis**: On-time performance by airport
- **Deep Dive (EDA)**: Exploratory data analysis with detailed insights
- **ML Prediction**: Interactive flight delay probability prediction
- **About**: Complete data science lifecycle documentation

## Installation

1. Clone the repository or download the files
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

### Data Source

The dataset contains 5.8 million US domestic flight records from 2015, sourced from the [US Department of Transportation via Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays).

### Data Format

The dashboard uses optimized Parquet format for fast loading:

- **Original**: ~500MB CSV file
- **Optimized**: 74MB Parquet file (85% reduction)

If you have the original `flights.csv`, it will be loaded automatically. For better performance, convert it to Parquet format using the included script.

## Machine Learning Model

### Option 1: Pre-train the Model (Recommended)

For faster dashboard loading, pre-train the model once:

```bash
python train_model.py
```

This will:
- Train a Random Forest classifier on the flight data
- Evaluate the model and display metrics
- Save the trained model to `flight_delay_model.pkl`

The dashboard will automatically load this pre-trained model for instant predictions.

### Option 2: Train On-the-Fly

If you don't pre-train the model, the dashboard will train it automatically on first load of the ML Prediction tab. This takes about 1-2 minutes but is cached for the session.

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure

```
├── app.py                      # Main Streamlit dashboard
├── utils.py                    # Data loading and ML utilities
├── train_model.py              # Standalone model training script
├── flights.parquet             # Optimized flight data
├── airlines.csv                # Airline reference data
├── airports.csv                # Airport reference data
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Data Science Lifecycle

This project demonstrates a complete data science workflow:

1. **Problem Definition**: Predict flight delay probability
2. **Data Collection**: 5.8M flight records from [US DOT via Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)
3. **Data Preprocessing**: CSV to Parquet optimization, data cleaning
4. **EDA**: Interactive visualizations across multiple dimensions
5. **Modeling**: Random Forest classifier with 8 features
6. **Evaluation**: Accuracy, Precision, Recall, F1-Score
7. **Deployment**: Interactive Streamlit dashboard

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: Airline, Origin/Destination Airports, Month, Day of Week, Day, Scheduled Departure, Distance
- **Target**: Binary classification (Delayed >15 min vs On-Time)
- **Training**: 80/20 train-test split with stratification
- **Performance**: See ML Prediction tab for live metrics

## Technical Optimizations

- **Parquet Format**: 85% size reduction with gzip compression
- **Categorical Dtypes**: Efficient memory usage for string columns
- **Model Caching**: Train once, use across sessions
- **Pre-computed Metrics**: Dashboard metrics calculated once and reused
- **Parallel Processing**: Multi-core Random Forest training

## Requirements

- Python 3.8+
- streamlit
- plotly
- pandas
- scikit-learn

## License

This project is for educational purposes demonstrating data science best practices.
