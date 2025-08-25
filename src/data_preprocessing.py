"""
Data preprocessing module for house price prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """Handles data preprocessing for house price prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prepare data for training and testing
        
        Args:
            df: pandas DataFrame with house data
            test_size: proportion of data for testing
            random_state: random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Feature engineering
        data = self._engineer_features(data)
        
        # Encode categorical variables
        data = self._encode_categorical(data)
        
        # Separate features and target
        X = data.drop('price', axis=1)
        y = data['price']
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale numerical features
        X_train_scaled = self._scale_features(X_train, fit=True)
        X_test_scaled = self._scale_features(X_test, fit=False)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Fill missing numerical values with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _engineer_features(self, df):
        """Create new features from existing ones"""
        # Price per square foot (if we have price - for training data)
        if 'price' in df.columns and 'area' in df.columns:
            df['price_per_sqft'] = df['price'] / df['area']
        
        # Total rooms
        if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
            df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        
        # Create age categories if we don't have age column
        # We'll create a synthetic age based on other features
        if 'age' not in df.columns:
            # Create a synthetic age based on features like furnishing status and mainroad
            # This is a simple heuristic - better to have actual age data
            df['age'] = 15  # Default age
            
            # Adjust based on furnishing status
            if 'furnishingstatus' in df.columns:
                df.loc[df['furnishingstatus'] == 'furnished', 'age'] = 10
                df.loc[df['furnishingstatus'] == 'semi-furnished', 'age'] = 15
                df.loc[df['furnishingstatus'] == 'unfurnished', 'age'] = 25
        
        # Age categories
        df['age_category'] = pd.cut(
            df['age'],
            bins=[0, 5, 15, 30, 100],
            labels=['New', 'Recent', 'Mature', 'Old']
        )
        
        # Bathroom to bedroom ratio
        if 'bathrooms' in df.columns and 'bedrooms' in df.columns:
            df['bath_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1)  # +1 to avoid division by zero
        
        # Size categories
        if 'area' in df.columns:
            df['size_category'] = pd.cut(
                df['area'],
                bins=[0, 3000, 6000, 9000, 15000, np.inf],
                labels=['Small', 'Medium', 'Large', 'Very Large', 'Mansion']
            )
        
        # Story categories
        if 'stories' in df.columns:
            df['story_category'] = df['stories'].apply(
                lambda x: 'Single' if x == 1 else 'Multi' if x <= 3 else 'High-rise'
            )
        
        # Premium features count
        premium_features = ['guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        df['premium_features_count'] = 0
        for feature in premium_features:
            if feature in df.columns:
                df['premium_features_count'] += (df[feature] == 'yes').astype(int)
        
        return df
    
    def _encode_categorical(self, df):
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle new categories during prediction
                known_categories = set(self.label_encoders[col].classes_)
                df[col] = df[col].astype(str)
                
                # Replace unknown categories with most frequent category
                unknown_mask = ~df[col].isin(known_categories)
                if unknown_mask.any():
                    most_frequent = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df.loc[unknown_mask, col] = most_frequent
                
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _scale_features(self, X, fit=True):
        """Scale numerical features"""
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled
    
    def prepare_data_for_prediction(self, df):
        """Transform new data using fitted preprocessors for prediction"""
        data = df.copy()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Engineer features
        data = self._engineer_features(data)
        
        # Remove price column if it exists (for prediction)
        if 'price' in data.columns:
            data = data.drop('price', axis=1)
        
        # Encode categorical variables
        data = self._encode_categorical(data)
        
        # Ensure all required columns are present
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in data.columns:
                    data[col] = 0  # Add missing columns with default value
            
            # Select only the required columns in the correct order
            data = data[self.feature_columns]
        
        # Scale features
        data_scaled = self._scale_features(data, fit=False)
        
        return data_scaled
    
    def transform_new_data(self, df):
        """Transform new data using fitted preprocessors - alias for prepare_data_for_prediction"""
        return self.prepare_data_for_prediction(df)
