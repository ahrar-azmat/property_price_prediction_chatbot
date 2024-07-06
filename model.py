import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def load_and_preprocess_data(file_path):
    logger.debug("Loading and preprocessing data")
    
    # Load data
    df = pd.read_csv(file_path)
    logger.debug("Data loaded successfully")
    
    # Handling missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    logger.debug("Handling missing values")
    
    # Encoding categorical variables
    df = pd.get_dummies(df, columns=['location', 'property_type'], drop_first=True)
    logger.debug("Encoding categorical variables")
    
    return df

# Load and preprocess the data
data = load_and_preprocess_data('property_data.csv')

# Feature columns
feature_columns = ['area', 'bedrooms', 'bathrooms', 'year_built', 'location_Location_B', 'location_Location_C', 'property_type_Condo', 'property_type_House']

# Split the data
X = data[feature_columns]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
prediction_model = LinearRegression()
logger.debug("Training Linear Regression model")
prediction_model.fit(X_train_scaled, y_train)

# Evaluate the model
logger.debug("Evaluating model")
y_pred = prediction_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

logger.info(f"MAE: {mae}")
logger.info(f"MSE: {mse}")
logger.info(f"RMSE: {rmse}")
logger.info(f"R2 Score: {r2}")

def predict_property_value(model, scaler, feature_columns, input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    
    # Scale the input data
    scaled_input = scaler.transform(input_df)
    
    # Predict the price
    predicted_price = model.predict(scaled_input)
    
    return predicted_price[0]

# Export the necessary components
__all__ = ['predict_property_value', 'feature_columns', 'prediction_model', 'scaler']
