# Crime Data Analysis and Prediction
This repo contains a notebook performing an extensive analysis of crime data, covering data loading, cleaning, exploratory data analysis (EDA), and the application of various machine learning models for forecasting, clustering, association rule mining, and classification.

# Author
Navdisha, Dhruv, and me.

# Dataset
This dataset comes from Toronto Police Service Public Safety Data Portal - https://data.torontopolice.on.ca/

Dataset - https://data.torontopolice.on.ca/datasets/0a239a5563a344a3bbf8452504ed8d68_0/explore

## Table of Contents
1.  [Data Loading](#data-loading)
2.  [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
3.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4.  [Machine Learning Models](#machine-learning-models)
    *   [1. LSTM for Time Series Forecasting](#1-lstm-for-time-series-forecasting)
    *   [2. DBSCAN for Spatial Clustering](#2-dbscan-for-spatial-clustering)
    *   [3. Apriori for Association Rule Mining](#3-apriori-for-association-rule-mining)
    *   [4. CART for Crime Classification](#4-cart-for-crime-classification)

## Data Loading
The crime dataset (`crime.csv`) is loaded into a pandas DataFrame. The notebook provides options to load the data either locally or directly from Google Drive.

## Data Cleaning and Preprocessing
The data cleaning process involves several key steps:
*   **Date Conversion**: `OCC_DATE` and `REPORT_DATE` columns are converted to datetime objects.
*   **Handling "NSA" values**: "NSA" (Not Stated/Applicable) values are replaced with `np.nan` for consistent missing value representation.
*   **Imputing Missing Dates**: Missing values in `OCC_YEAR`, `OCC_MONTH`, `OCC_DAY`, `OCC_DOY`, `OCC_DOW` are filled using information extracted from the `OCC_DATE` column.
*   **Imputing Categorical Values**: Missing categorical values (e.g., `DIVISION`, `HOOD_158`) are filled with 'Unknown'.
*   **Filtering Outliers**: Rows where `REPORT_DATE` and `OCC_DATE` differ by more than 20 years are removed.
*   **Handling Small Numerical Values**: Exceptionally small numerical values (less than 1e-3) in numerical columns are replaced with `np.nan` and then imputed with the column's median to ensure data integrity.
*   **Feature Engineering**:
    *   **SEASON**: A 'SEASON' column is created by mapping `OCC_MONTH` to Winter, Spring, Summer, or Fall.
    *   **Geo_Region**: The dataset is divided into five geographical regions (Central, North, South, East, West) based on latitude and longitude quantiles.

## Exploratory Data Analysis (EDA)
Several visualizations are generated to understand the crime data:
1.  **Hourly Crime Trends Throughout the Week**: A heatmap showing the frequency of crimes by hour of the day and day of the week.
2.  **Crime Category Distribution by Premises Type**: Bar plots illustrating the distribution of crime categories across different premises types, both in raw counts and normalized percentages.
3.  **Crime Count by Premises Type and Geo Region**: A grouped bar chart showing crime counts segmented by premises type and the engineered geographical regions.
4.  **Top 20 Crime Types**: A horizontal bar chart displaying the most frequent crime offenses.

## Machine Learning Models
The notebook implements four different machine learning models:

### 1. LSTM for Time Series Forecasting
*   **Objective**: Predict future weekly crime counts based on historical data.
*   **Preprocessing**: Crime data is grouped by `PREMISES_TYPE` and `OCC_DATE` (weekly frequency). Additional time-based features (`day_of_week`, `month`, `day_of_year`, `is_weekend`) are extracted. Categorical `PREMISES_TYPE` is one-hot encoded. Data is scaled using `MinMaxScaler`.
*   **Model**: A Sequential LSTM model is built with two LSTM layers and Dense layers, compiled with `adam` optimizer and `mse` loss.
*   **Evaluation**: The model's performance is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on the test set. A plot comparing actual vs. predicted values is generated.

### 2. DBSCAN for Spatial Clustering
*   **Objective**: Identify spatial clusters of crime incidents.
*   **Preprocessing**: A 20% sample of the data is used to prevent memory issues. Only `LONG_WGS84`, `LAT_WGS84`, and `MCI_CATEGORY` are used. Latitude and longitude are scaled using `StandardScaler`.
*   **Model**: DBSCAN algorithm is applied to the scaled coordinates. `eps` is set to 0.25 and `min_samples` to 60 to define clusters.
*   **Visualization**: The spatial distribution of the top 4 crime types (Assault, Break and Enter, Auto Theft, Robbery) is visualized using a scatter plot, showing distinct crime patterns.

### 3. Apriori for Association Rule Mining
*   **Objective**: Discover associations between crime types, locations, and times.
*   **Preprocessing**: Features include `MCI_CATEGORY` (Crime), `PREMISES_TYPE` (Location), `OCC_HOUR` (Time of Day), and `OCC_DOW` (Day of Week). These are one-hot encoded into a basket format.
*   **Model**: The Apriori algorithm generates frequent itemsets, and then association rules are derived using a `min_support` of 0.01 and `min_threshold` (lift) of 1.1.
*   **Analysis**: The strongest rules for target crimes (Assault, Robbery, Break and Enter, Auto Theft) are identified and interpreted.
*   **Visualization**: A scatter plot shows the relationship between support, confidence, and lift for all generated rules. A heatmap (Risk Matrix) and a Network Graph (Crime Web) are used for advanced visualization of crime associations.

### 4. CART for Crime Classification
*   **Objective**: Classify the `MCI_CATEGORY` of a crime based on features like time, location, and premises type.
*   **Preprocessing**: Features include `OCC_YEAR`, `OCC_MONTH`, `OCC_DOW`, `OCC_HOUR`, `PREMISES_TYPE`, and `DIVISION`. Categorical features and the target `MCI_CATEGORY` are label encoded. 'Theft Over' crimes are excluded from the analysis.
*   **Model**: A Decision Tree Classifier (`DecisionTreeClassifier`) is trained with `max_depth=10` and `min_samples_leaf=50`.
*   **Evaluation**: Model performance is assessed using accuracy, F1-score, and a classification report. A confusion matrix, feature importances, and a visualization of the decision tree structure (top 3 levels) are provided.
*   **Cross-Validation**: K-Fold Cross-Validation (5-fold stratified) is implemented to prove the model's validity and accuracy across different data subsets, providing average metrics for robustness.
