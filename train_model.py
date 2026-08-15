import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils.housing_data_preprocessor import HousingDataPreprocessor

class HousingMarketModel:
    def __init__(self):
        self.model = None
        self.preprocessor = HousingDataPreprocessor()
        self.best_model_name = None
        
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define models to test
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        # Train and evaluate models
        best_score = -np.inf
        best_model = None
        best_model_name = None
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # For ensemble methods, do hyperparameter tuning
            if name == 'Random Forest':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, 
                    scoring='r2', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
            elif name == 'Gradient Boosting':
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
                grid_search = GridSearchCV(
                    model, param_grid, cv=5,
                    scoring='r2', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
            else:
                # TRAIN LINEAR MODEL (no tuning needed)
                model.fit(X_train, y_train)
            
            # Make predictions # EVALUATE
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'R2': r2,
                'model': model
            }
            
            print(f"{name} - R2 Score: {r2:.4f}, MAE: ${mae:,.2f}")
            
            # Select best model
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name
        
        self.model = best_model
        self.best_model_name = best_model_name
        
        # Print summary
        print("\n" + "="*50)
        print(f"Best Model: {best_model_name}")
        print(f"R2 Score: {best_score:.4f}")
        print("="*50)
        
        # Plot feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            self.plot_feature_importance(best_model)
        
        return results, X_train, X_test, y_train, y_test
    
    def plot_feature_importance(self, model):
        """Plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.preprocessor.feature_columns
            
            # Create DataFrame for plotting
            feat_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10))
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.show()
    
    def save_model(self, model_path, preprocessor_path):
        """Save the trained model and preprocessor"""
        joblib.dump(self.model, model_path)
        self.preprocessor.save_preprocessor(preprocessor_path)
        print(f"Model saved to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")
    
    def load_model(self, model_path, preprocessor_path):
        """Load the trained model and preprocessor"""
        self.model = joblib.load(model_path)
        self.preprocessor.load_preprocessor(preprocessor_path)
        print("Model and preprocessor loaded successfully")
    
    def predict(self, features_predict):
        """Make a prediction for a single house"""

        # VALIDATE MODEL IS READY
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess the features
        X_scaled = self.preprocessor.preprocess_single(features_predict)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)
        
        return prediction[0]

def main():
    # Initialize preprocessor
    preprocessor = HousingDataPreprocessor()
    
    # Load and preprocess data
    print("Loading data...")
    df = preprocessor.load_data('data/ontario_housing.csv')
    print(f"Data loaded: {len(df)} samples")
    
    print("\nPreprocessing data...")
    X, y = preprocessor.preprocess(df)
    print(f"Features shape: {X.shape}")
    
    # Train model
    print("\nTraining models...")
    model_trainer = HousingMarketModel()
    model_trainer.preprocessor = preprocessor  # Share preprocessor
    
    results, X_train, X_test, y_train, y_test = model_trainer.train_models(X, y)
    
    # Save model
    print("\nSaving model...")
    model_trainer.save_model('models/model.pkl', 'models/preprocessor.pkl')
    
    # Save preprocessor for later use
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    print("\n✅ Model training complete!")

if __name__ == "__main__":
    main()