# 🏠 House Price Predictor

A desktop machine learning application that estimates house prices from property characteristics such as size, bedrooms, bathrooms, age, location, garage availability, and condition.

The project demonstrates an end-to-end supervised machine learning workflow: data preparation, categorical encoding, feature scaling, train/test splitting, model comparison, hyperparameter tuning, evaluation, model persistence, and prediction through a Tkinter graphical interface.

## ✨ Features

- House-price prediction from seven property features
- Input validation for numeric fields and supported ranges
- Data preview and descriptive-statistics views
- Feature-importance and price-vs-size visualizations
- Training and comparison of multiple regression algorithms
- Hyperparameter tuning with GridSearchCV
- Model evaluation using R², MAE, and MSE
- Automatic selection of the model with the highest test-set R²
- Saved model and preprocessing artifacts using joblib
- Reusable preprocessing for both training data and new predictions

## Tech Stack

- **Python 3.8+**
- **Machine Learning**: scikit-learn, pandas, numpy
- **UI Framework**: tkinter
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib

## 🧠 Machine Learning Workflow

``` text
Housing Dataset
      │
      ▼
Data Preprocessing
      │
      ├── Missing-value handling
      ├── Categorical encoding
      └── Feature scaling
      │
      ▼
Train / Test Split
      │
      ▼
Model Training
      │
      ├── Linear Regression
      ├── Random Forest Regressor
      └── Gradient Boosting Regressor
      │
      ▼
Hyperparameter Tuning
(Random Forest & Gradient Boosting)
      │
      ▼
Model Evaluation
(R² / MAE / MSE)
      │
      ▼
Best Model Selection
      │
      ▼
Save Model + Preprocessor
      │
      ▼
Tkinter Prediction Application
```
# 🤖 Models

- The training pipeline compares three models.
- Random Forest and Gradient Boosting are tuned with `GridSearchCV` using 5-fold cross-validation. 
- The final model is selected using the highest R² score on the held-out test set.

 **Current Evaluation Results**

Using the included 1,000-row dataset and the current `random_state=42` train/test split:

**Model**                     **R²**        **MAE**
----------------------- ------------ -----------------
Linear Regression             0.7431       $34,186.76
Random Forest                 0.7882       $31,542.90
Gradient Boosting             0.8213       $28,863.14

In the current configuration, **Gradient Boosting** produced the strongest test-set R².
These metrics describe this project dataset and split only; 
they should not be interpreted as real-world housing-market accuracy.

## 🏡 Prediction Features

Feature             Description
------------------- -----------------------------------
`size_sqft`         Property size in square feet
`total_bedrooms`    Number of bedrooms
`total_bathrooms`   Number of bathrooms
`house_age`         Property age in years
`location`          Urban, Suburban, or Rural
`garage`            Garage availability
`condition`         Excellent, Good, Average, or Poor

The target variable is `price`.

## 🧹 Data Preprocessing

`HousingDataPreprocessor` is responsible for preparing data for model training and inference.

The current pipeline:
- loads the housing dataset with pandas
- removes rows containing missing values
- separates predictors from the `price` target
- encodes categorical columns with `LabelEncoder`
- scales features with `StandardScaler`
- stores feature order, encoders, and scaler for future predictions

The fitted preprocessing objects are saved alongside the trained model
so new user inputs are transformed consistently before prediction.

## 🖥️ Desktop Application
The Tkinter application contains three main tabs.

### 🔮 Predict 
Users enter property details and receive an estimated price from the saved model.

### 📊 Data 
Displays a sample of the housing dataset together with descriptive statistics.

### 📈 Visualizations 
Displays feature importance (when available for the selected model) and a scatter plot showing house size versus price.

## 📁 Project Structure
``` text
house-price-predictor/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── feature_importance.png
│
├── data/
│   └── ontario_housing.csv
│
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
│
└── utils/
    └── housing_data_preprocessor.py
```
## 🛠️ Technology Stack

- **Python**
- **pandas** --- data loading and manipulation
- **NumPy** --- numerical operations
- **scikit-learn** --- preprocessing, regression, tuning, and evaluation
- **Matplotlib / Seaborn** --- visualizations
- **Tkinter** --- desktop graphical user interface
- **joblib** --- model and preprocessor persistence

## 🚀 Installation

### 1. Clone the repository:
```bash
git clone https://github.com/soniakataria10/housing-market-predictors.git
cd housing-market-predictors
```
### 2. Create a virtual environment:
```bash
python -m venv .venv
```
### 3. Activate it
Windows PowerShell:
```bash
.\.venv\Scripts\Activate.ps1
```
### 4. Install dependencies:
```bash
pip install -r requirements.txt
```
> Tkinter is normally distributed with standard Python installations
> rather than installed from PyPI. On some Linux distributions it must
> be installed through the operating system package manager.

## 🏋️ Train the Model

Run:
```bash
python train_model.py
```
The training script:
1.  loads and preprocesses the dataset
2.  creates an 80/20 train/test split
3.  trains three regression algorithms
4.  tunes the ensemble models
5.  evaluates each model
6.  selects the best model by R²
7.  saves the trained model and fitted preprocessor

Generated artifacts are stored under `models/`.

## ▶️ Run the Application

After training the model:
```bash
python app.py
```
Enter property details in the **Predict** tab and select **Predict Price**.

## 📊 Evaluation Metrics

The project evaluates regression models using:

- **R² (Coefficient of Determination)** --- measures how much variance in house prices is explained by the model.
- **MAE (Mean Absolute Error)** --- measures the average absolute prediction error in price units.
- **MSE (Mean Squared Error)** --- penalizes larger prediction errors more heavily.

## ⚠️ Limitations

-   The included dataset is suitable for demonstrating the ML workflow
    but should not be treated as a production real-estate valuation
    dataset.
-   Location is represented only by broad categories (`Urban`,
    `Suburban`, `Rural`) rather than real geographic features.
-   The model does not include market-time variables such as interest
    rates, sale date, neighborhood trends, or comparable sales.
-   Label encoding imposes numeric codes on categorical values;
    production systems would typically evaluate encodings such as
    one-hot encoding for nominal features.
-   The current desktop application is intended for local, single-user
    execution.
-   Prediction uncertainty is not currently calibrated, so the UI should
    not treat a fixed confidence label as a statistical confidence
    interval.

