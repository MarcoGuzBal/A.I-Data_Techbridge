import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load cleaned data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the data for regression task."""
    # Separate features (independent variables) and target (dependent variable 'TIME')
    X = df.drop(columns=['TIME'])  # Drop 'TIME' column for features
    y = df['TIME']  # 'TIME' as the dependent variable
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns

def feature_selection(X_train, y_train, feature_names, n_features=10, verbose=1):
    """Select the most important features using Random Forest importance."""
    print("Performing feature selection...")
    
    # Train a small random forest for feature selection
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf_selector.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    if verbose:
        print("Feature ranking:")
        for i in range(min(20, X_train.shape[1])):  # Show top 20 features
            if i < len(feature_names):
                print(f"{i+1}. Feature {indices[i]} ({feature_names[indices[i]] if indices[i] < len(feature_names) else 'unknown'}) - {importances[indices[i]]:.4f}")
            else:
                print(f"{i+1}. Feature {indices[i]} - {importances[indices[i]]:.4f}")
    
    # Create feature selector
    selector = SelectFromModel(rf_selector, threshold=-np.inf, max_features=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # Get indices of selected features
    selected_features = np.where(selector.get_support())[0]
    print(f"Selected feature indices: {selected_features}")
    if len(feature_names) > 0:
        selected_names = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in selected_features]
        print(f"Selected feature names: {selected_names}")
    
    return X_train_selected, selector

def train_random_forest_model(X_train, y_train, params=None, verbose=1):
    """Train a Random Forest regression model."""
    print("Training Random Forest regression model...")
    
    # Default parameters if none provided
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto',
            'bootstrap': True
        }
    
    # Create and train the model
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        bootstrap=params['bootstrap'],
        random_state=42,
        n_jobs=-1,
        verbose=verbose
    )
    
    # Perform cross-validation before final training
    if verbose:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean CV R² score: {cv_scores.mean():.4f}")
    
    # Train final model on all training data
    model.fit(X_train, y_train)
    
    return model

def evaluate_regression_model(model, X_test, y_test, visualize=True):
    """Evaluate the regression model using appropriate metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    print("\nRegression Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Add more detailed analysis
    error = y_pred - y_test
    print(f"\nError Analysis:")
    print(f"Mean Error: {np.mean(error):.4f}")
    print(f"Error Standard Deviation: {np.std(error):.4f}")
    print(f"Median Error: {np.median(error):.4f}")
    print(f"Min/Max Error: {np.min(error):.4f}/{np.max(error):.4f}")
    
    # Calculate percentage of predictions within certain error ranges
    within_5_percent = np.mean(np.abs(error) <= 0.05 * np.abs(y_test)) * 100
    within_10_percent = np.mean(np.abs(error) <= 0.10 * np.abs(y_test)) * 100
    within_20_percent = np.mean(np.abs(error) <= 0.20 * np.abs(y_test)) * 100
    
    print(f"\nPrediction Accuracy:")
    print(f"Predictions within 5% of actual value: {within_5_percent:.2f}%")
    print(f"Predictions within 10% of actual value: {within_10_percent:.2f}%")
    print(f"Predictions within 20% of actual value: {within_20_percent:.2f}%")
    
    # Visualize results
    if visualize:
        plt.figure(figsize=(12, 10))
        
        # Actual vs Predicted Plot
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted')
        
        # Residuals Plot
        plt.subplot(2, 2, 2)
        plt.scatter(y_pred, error, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # Histogram of Errors
        plt.subplot(2, 2, 3)
        plt.hist(error, bins=30, alpha=0.7)
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.subplot(2, 2, 4)
            num_features = min(X_test.shape[1], 10)  # Show at most 10 features
            plt.barh(range(num_features), importances[indices[:num_features]], align='center')
            plt.yticks(range(num_features), [f'Feature {indices[i]}' for i in range(num_features)])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
        else:
            # QQ Plot as alternative if feature_importances_ not available
            plt.subplot(2, 2, 4)
            import scipy.stats as stats
            stats.probplot(error, plot=plt)
            plt.title('Q-Q Plot of Errors')
        
        plt.tight_layout()
        plt.savefig('rf_regression_evaluation.png')
        print("\nEvaluation plots saved to 'rf_regression_evaluation.png'")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

def save_model(model, scaler, selector, model_filename, scaler_filename, selector_filename):
    """Save the trained model, scaler and selector to disk."""
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(selector, selector_filename)
    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")
    print(f"Selector saved to {selector_filename}")

def main():
    parser = argparse.ArgumentParser(description="Train a Random Forest regression model.")
    parser.add_argument('file_path', type=str, help="Path to the cleaned CSV file.")
    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument('--max_depth', type=int, default=None, help="Maximum depth of trees.")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Minimum samples to split a node.")
    parser.add_argument('--min_samples_leaf', type=int, default=1, help="Minimum samples in a leaf node.")
    parser.add_argument('--model_filename', type=str, default='rf_regression_model.pkl', 
                        help="Filename for saving the trained model.")
    parser.add_argument('--scaler_filename', type=str, default='scaler.pkl', 
                        help="Filename for saving the feature scaler.")
    parser.add_argument('--selector_filename', type=str, default='selector.pkl', 
                        help="Filename for saving the feature selector.")
    parser.add_argument('--n_features', type=int, default=10, help="Number of features to select.")
    parser.add_argument('--verbose', type=int, default=1, help="Verbosity level (0=silent, 1=progress).")
    parser.add_argument('--no_plot', action='store_true', help="Disable visualization plots.")
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.file_path)
    
    # Preprocess data
    X_scaled, y, scaler, feature_names = preprocess_data(df)
    
    # Split data before feature selection to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Feature selection only on training data
    X_train_selected, selector = feature_selection(X_train, y_train, feature_names, args.n_features, args.verbose)
    
    # Apply the same feature selection to test data
    X_test_selected = selector.transform(X_test)
    
    # Prepare RF parameters
    rf_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'max_features': 'sqrt',  # 'sqrt' generally works well for RF
        'bootstrap': True
    }
    
    # Train the Random Forest regression model
    model = train_random_forest_model(X_train_selected, y_train, rf_params, args.verbose)
    
    # Evaluate the model
    evaluate_regression_model(model, X_test_selected, y_test, not args.no_plot)
    
    # Save the model, scaler and selector to disk
    save_model(model, scaler, selector, args.model_filename, args.scaler_filename, args.selector_filename)

if __name__ == "__main__":
    main()