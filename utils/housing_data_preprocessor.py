import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class HousingDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'price'
        
    def load_data(self, filepath):
        """Load housing market dataset"""
        try:
            # Create sample data if file doesn't exist
            try:
                df = pd.read_csv(filepath)
            except FileNotFoundError:
                df = self.create_sample_data()
                df.to_csv(filepath, index=False)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def create_sample_data(self):
        """Generate complete dataset with location feature"""
        np.random.seed(42)
        n_samples = 1000
        # Base features
        size_sqft = np.random.normal(1500, 400, n_samples).clip(600, 3000)
        bedrooms = np.random.choice([1,2,3,4,5], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15])
        bathrooms = np.where(
            bedrooms <= 2,
            np.random.choice([1, 1.5, 2], n_samples, p=[0.3, 0.4, 0.3]),
            np.random.choice([2, 2.5, 3, 3.5], n_samples, p=[0.2, 0.3, 0.3, 0.2])
        )
        age = np.random.exponential(15, n_samples).clip(0, 50).astype(int)
        
        # LOCATION with realistic distribution
        location = np.random.choice(
            ['Urban', 'Suburban', 'Rural'], 
            n_samples, 
            p=[0.4, 0.4, 0.2]  # 40% Urban, 40% Suburban, 20% Rural
        )
        
        garage = np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35])
        condition = np.random.choice(
            ['Excellent', 'Good', 'Average', 'Poor'], 
            n_samples, 
            p=[0.40, 0.25, 0.20, 0.15]
        )
        
        # Price calculation with LOCATION EFFECT
        base_price = 300000
        
        # Size effect
        size_effect = size_sqft * 150
        
        # Bedroom effect
        bedroom_effect = bedrooms * 12000
        
        # Bathroom effect
        bathroom_effect = bathrooms * 15000
        
        # Age effect (NEGATIVE - older is cheaper)
        age_effect = -age * 1000
        
        # LOCATION EFFECT (Clear differences)
        location_effects = {
            'Urban': 50000,      # Urban premium
            'Suburban': 20000,   # Suburban moderate
            'Rural': -20000      # Rural discount
        }
        location_effect = [location_effects[loc] for loc in location]
        
        # Garage effect
        garage_effect = np.where(np.array(garage) == 'Yes', 20000, 0)
        
        # Condition effects
        condition_effects = {
            'Excellent': 35000,
            'Good': 15000,
            'Average': -10000,
            'Poor': -40000
        }
        condition_effect = [condition_effects[c] for c in condition]
        
        # Random noise
        noise = np.random.normal(0, 30000, n_samples)
        
        # Calculate final price
        price = (
            base_price + 
            size_effect + 
            bedroom_effect + 
            bathroom_effect + 
            age_effect + 
            location_effect + 
            garage_effect + 
            condition_effect + 
            noise
        )
        
        # Clean up
        price = price.clip(50000, 1200000)
        price = np.round(price / 1000) * 1000
        
        # Create DataFrame
        df = pd.DataFrame({
            'size_sqft': np.round(size_sqft).astype(int),
            'total_bedrooms': bedrooms,
            'total_bathrooms': bathrooms,
            'house_age': age,
            'location': location,
            'garage': garage,
            'condition': condition,
            'price': price.astype(int)
        })
        
        return df
    
    def preprocess(self, df):
        """Preprocess the data"""
        if df is None:
            return None, None
        
        # Handle missing values
        
        # Option 1: Drop rows with missing values (current approach)
        df = df.dropna()

        # Option 2: Fill with mean/median (for numerical features)
        df['size_sqft'] = df['size_sqft'].fillna(df['size_sqft'].mean())
        df['price'] = df['price'].fillna(df['price'].median())

        # Option 3: Fill with mode (for categorical features)
        df['location'] = df['location'].fillna(df['location'].mode()[0])

        # Option 4: Fill with specific value
        df['house_age'] = df['house_age'].fillna(0)

        # Option 5: Forward fill (for time series data)
        df = df.fillna(method='ffill')

        # Option 6: Interpolation
        df = df.interpolate()

        # Separate features and target
        X = df.drop(self.target_column, axis=1)
        # X = df.drop(self.feature_columns, axis=1)
        y = df[self.target_column]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Encode categorical variables
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            self.label_encoders[column] = le
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def preprocess_single(self, features_predict):
        """Preprocess a single prediction input"""
        # Create DataFrame from input
        df = pd.DataFrame([features_predict])
        
        # Ensure all features are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Encode categorical variables
        for column in df.select_dtypes(include=['object']).columns:
            if column in self.label_encoders:
                le = self.label_encoders[column]
                # Handle unknown categories
                df[column] = df[column].map(lambda x: x if x in le.classes_ else le.classes_[0])
                df[column] = le.transform(df[column])
        
        # Scale features
        X_scaled = self.scaler.transform(df[self.feature_columns])
        return X_scaled
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor"""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }, filepath)
    
    def load_preprocessor(self, filepath):
        """Load the preprocessor"""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_columns = data['feature_columns']