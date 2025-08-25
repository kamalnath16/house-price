"""
Prediction module for house price prediction
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Union
import os

class HousePricePredictor:
    """Handles house price predictions using trained models"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        
    def load_model(self, model_path: str, preprocessor_path: str = None):
        """
        Load trained model and preprocessor
        
        Args:
            model_path: Path to the saved model
            preprocessor_path: Path to the saved preprocessor
        """
        try:
            self.model = joblib.load(model_path)
            if preprocessor_path and os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
            self.is_loaded = True
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.is_loaded = False
    
    def _prepare_features(self, house_features: Dict[str, Union[int, float, str]]) -> pd.DataFrame:
        """
        Prepare and validate input features to match training data format
        
        Args:
            house_features: Dictionary with house features
            
        Returns:
            DataFrame with prepared features
        """
        # Create a copy to avoid modifying original data
        features = house_features.copy()
        
        # Ensure all required features are present with default values based on your CSV columns
        required_features = {
            'price': 0,  # Will be removed, just needed for preprocessing
            'area': features.get('area', 2000),
            'bedrooms': features.get('bedrooms', 3),
            'bathrooms': features.get('bathrooms', 2),
            'stories': features.get('stories', 2),
            'mainroad': features.get('mainroad', 'yes'),
            'guestroom': features.get('guestroom', 'no'),
            'basement': features.get('basement', 'no'),
            'hotwaterheating': features.get('hotwaterheating', 'no'),
            'airconditioning': features.get('airconditioning', 'yes'),
            'parking': features.get('parking', 2),
            'prefarea': features.get('prefarea', 'no'),
            'furnishingstatus': features.get('furnishingstatus', 'semi-furnished')
        }
        
        # Update with actual values
        for key, value in features.items():
            if key in required_features:
                required_features[key] = value
        
        # Convert to DataFrame
        df = pd.DataFrame([required_features])
        
        return df
    
    def predict_single(self, house_features: Dict[str, Union[int, float, str]]) -> float:
        """
        Predict price for a single house
        
        Args:
            house_features: Dictionary with house features
            
        Returns:
            Predicted price
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Use load_model() first.")
        
        # Prepare features
        df = self._prepare_features(house_features)
        
        # Preprocess if preprocessor is available
        if self.preprocessor:
            df_processed = self.preprocessor.prepare_data_for_prediction(df)
        else:
            # Basic preprocessing without fitted preprocessor
            df_processed = self._basic_preprocessing(df)
        
        # Make prediction
        prediction = self.model.predict(df_processed)[0]
        
        return max(0, prediction)  # Ensure non-negative price
    
    def _basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic preprocessing when no preprocessor is available
        
        Args:
            df: Input dataframe
            
        Returns:
            Processed dataframe
        """
        processed_df = df.copy()
        
        # Remove price column if exists
        if 'price' in processed_df.columns:
            processed_df = processed_df.drop('price', axis=1)
        
        # Add synthetic age column if missing
        if 'age' not in processed_df.columns:
            processed_df['age'] = 15  # Default age
            if 'furnishingstatus' in processed_df.columns:
                processed_df.loc[processed_df['furnishingstatus'] == 'furnished', 'age'] = 10
                processed_df.loc[processed_df['furnishingstatus'] == 'semi-furnished', 'age'] = 15
                processed_df.loc[processed_df['furnishingstatus'] == 'unfurnished', 'age'] = 25
        
        # Engineer features
        if 'area' in processed_df.columns:
            # Price per sqft (we'll set a dummy value)
            processed_df['price_per_sqft'] = 0  # This will be calculated during training
        
        if 'bedrooms' in processed_df.columns and 'bathrooms' in processed_df.columns:
            processed_df['total_rooms'] = processed_df['bedrooms'] + processed_df['bathrooms']
            processed_df['bath_bed_ratio'] = processed_df['bathrooms'] / (processed_df['bedrooms'] + 1)
        
        # Age categories
        processed_df['age_category'] = pd.cut(
            processed_df['age'],
            bins=[0, 5, 15, 30, 100],
            labels=[0, 1, 2, 3]  # Use numeric labels for consistency
        ).astype(float)
        
        # Size categories
        if 'area' in processed_df.columns:
            processed_df['size_category'] = pd.cut(
                processed_df['area'],
                bins=[0, 3000, 6000, 9000, 15000, np.inf],
                labels=[0, 1, 2, 3, 4]  # Use numeric labels
            ).astype(float)
        
        # Story categories
        if 'stories' in processed_df.columns:
            processed_df['story_category'] = processed_df['stories'].apply(
                lambda x: 0 if x == 1 else 1 if x <= 3 else 2
            )
        
        # Premium features count
        premium_features = ['guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        processed_df['premium_features_count'] = 0
        
        # Handle categorical variables - simple encoding
        categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                          'airconditioning', 'prefarea', 'furnishingstatus']
        
        for col in categorical_cols:
            if col in processed_df.columns:
                if col == 'furnishingstatus':
                    # Map furnishing status
                    furnish_map = {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}
                    processed_df[col] = processed_df[col].map(furnish_map)
                else:
                    # Map yes/no to 1/0 and count premium features
                    processed_df[col] = processed_df[col].map({'yes': 1, 'no': 0})
                    if col in premium_features:
                        processed_df['premium_features_count'] += processed_df[col]
        
        # Ensure all values are numeric
        processed_df = processed_df.apply(pd.to_numeric, errors='coerce')
        
        # Fill any NaN values with 0
        processed_df = processed_df.fillna(0)
        
        return processed_df
    
    def predict_batch(self, houses_list: List[Dict[str, Union[int, float, str]]]) -> List[float]:
        """
        Predict prices for multiple houses
        
        Args:
            houses_list: List of dictionaries with house features
            
        Returns:
            List of predicted prices
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Use load_model() first.")
        
        predictions = []
        for house in houses_list:
            try:
                prediction = self.predict_single(house)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error predicting for house {house}: {e}")
                predictions.append(0)  # Default value for failed predictions
        
        return predictions
    
    def predict_from_csv(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Predict prices from a CSV file
        
        Args:
            csv_path: Path to CSV file with house features
            output_path: Path to save results (optional)
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Use load_model() first.")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Convert each row to dictionary and predict
        predictions = []
        for _, row in df.iterrows():
            house_dict = row.to_dict()
            try:
                prediction = self.predict_single(house_dict)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error predicting for row {row.name}: {e}")
                predictions.append(0)
        
        # Add predictions to original DataFrame
        result_df = df.copy()
        result_df['predicted_price'] = predictions
        result_df['formatted_price'] = result_df['predicted_price'].apply(lambda x: f"${x:,.2f}")
        
        # Save if output path is provided
        if output_path:
            result_df.to_csv(output_path, index=False)
            print(f"📁 Predictions saved to {output_path}")
        
        return result_df
    
    def get_prediction_confidence(self, house_features: Dict[str, Union[int, float, str]], 
                                n_estimations: int = 100) -> Dict[str, float]:
        """
        Get prediction with confidence intervals (for ensemble models)
        
        Args:
            house_features: Dictionary with house features
            n_estimations: Number of estimations for confidence interval
            
        Returns:
            Dictionary with prediction and confidence intervals
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Use load_model() first.")
        
        # Check if model supports prediction intervals
        if hasattr(self.model, 'estimators_'):
            # For Random Forest, use individual trees
            df = self._prepare_features(house_features)
            
            if self.preprocessor:
                df_processed = self.preprocessor.prepare_data_for_prediction(df)
            else:
                df_processed = self._basic_preprocessing(df)
            
            # Get predictions from individual trees
            tree_predictions = []
            for estimator in self.model.estimators_[:n_estimations]:
                pred = estimator.predict(df_processed)[0]
                tree_predictions.append(max(0, pred))
            
            predictions_array = np.array(tree_predictions)
            
            return {
                'prediction': predictions_array.mean(),
                'lower_bound': np.percentile(predictions_array, 2.5),
                'upper_bound': np.percentile(predictions_array, 97.5),
                'std': predictions_array.std(),
                'confidence_interval': f"${np.percentile(predictions_array, 2.5):,.0f} - ${np.percentile(predictions_array, 97.5):,.0f}"
            }
        else:
            # For non-ensemble models, just return the prediction
            prediction = self.predict_single(house_features)
            return {
                'prediction': prediction,
                'lower_bound': prediction,
                'upper_bound': prediction,
                'std': 0.0,
                'confidence_interval': f"${prediction:,.0f}"
            }
    
    def explain_prediction(self, house_features: Dict[str, Union[int, float, str]]) -> Dict[str, Dict[str, float]]:
        """
        Explain prediction using feature importance (for tree-based models)
        
        Args:
            house_features: Dictionary with house features
            
        Returns:
            Dictionary with feature contributions
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Use load_model() first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            return {"message": "Model doesn't support feature importance explanation"}
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        
        # Convert features to DataFrame and preprocess
        df = self._prepare_features(house_features)
        
        if self.preprocessor:
            df_processed = self.preprocessor.prepare_data_for_prediction(df)
        else:
            df_processed = self._basic_preprocessing(df)
        
        # Calculate feature contributions
        feature_values = df_processed.iloc[0].values
        contributions = feature_importance * np.abs(feature_values)
        
        # Normalize contributions
        total_contribution = contributions.sum()
        if total_contribution > 0:
            normalized_contributions = contributions / total_contribution
        else:
            normalized_contributions = contributions
        
        # Create explanation dictionary
        explanation = {}
        for i, feature_name in enumerate(df_processed.columns):
            explanation[feature_name] = {
                'importance': float(feature_importance[i]),
                'value': float(feature_values[i]),
                'contribution': float(normalized_contributions[i]),
                'contribution_percent': f"{normalized_contributions[i] * 100:.1f}%"
            }
        
        return explanation
    
    def get_similar_houses(self, house_features: Dict[str, Union[int, float, str]], 
                          reference_data: pd.DataFrame, n_similar: int = 5) -> pd.DataFrame:
        """
        Find similar houses in the reference dataset
        
        Args:
            house_features: Dictionary with house features
            reference_data: Reference dataset to search in
            n_similar: Number of similar houses to return
            
        Returns:
            DataFrame with similar houses
        """
        try:
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler
            
            # Preprocess both target house and reference data
            target_df = self._prepare_features(house_features)
            
            if self.preprocessor:
                target_processed = self.preprocessor.prepare_data_for_prediction(target_df)
                reference_processed = self.preprocessor.prepare_data_for_prediction(reference_data)
            else:
                target_processed = self._basic_preprocessing(target_df)
                reference_processed = self._basic_preprocessing(reference_data)
            
            # Ensure same columns
            common_columns = list(set(target_processed.columns) & set(reference_processed.columns))
            target_processed = target_processed[common_columns]
            reference_processed = reference_processed[common_columns]
            
            # Scale the data
            scaler = StandardScaler()
            reference_scaled = scaler.fit_transform(reference_processed)
            target_scaled = scaler.transform(target_processed)
            
            # Find nearest neighbors
            nn_model = NearestNeighbors(n_neighbors=min(n_similar, len(reference_data)), 
                                     metric='euclidean')
            nn_model.fit(reference_scaled)
            
            distances, indices = nn_model.kneighbors(target_scaled)
            
            # Get similar houses
            similar_houses = reference_data.iloc[indices[0]].copy()
            similar_houses['distance'] = distances[0]
            similar_houses['similarity_score'] = 1 / (1 + distances[0])
            
            return similar_houses.sort_values('distance')
            
        except ImportError:
            return pd.DataFrame({"error": ["sklearn not available for similarity search"]})
        except Exception as e:
            return pd.DataFrame({"error": [f"Error in similarity search: {str(e)}"]})


def create_sample_predictions():
    """Create sample prediction examples matching your dataset structure"""
    sample_houses = [
        {
            'area': 7420,
            'bedrooms': 4,
            'bathrooms': 2,
            'stories': 3,
            'mainroad': 'yes',
            'guestroom': 'no',
            'basement': 'no',
            'hotwaterheating': 'no',
            'airconditioning': 'yes',
            'parking': 2,
            'prefarea': 'yes',
            'furnishingstatus': 'furnished'
        },
        {
            'area': 8960,
            'bedrooms': 4,
            'bathrooms': 4,
            'stories': 4,
            'mainroad': 'yes',
            'guestroom': 'no',
            'basement': 'no',
            'hotwaterheating': 'no',
            'airconditioning': 'yes',
            'parking': 3,
            'prefarea': 'no',
            'furnishingstatus': 'furnished'
        },
        {
            'area': 3500,
            'bedrooms': 2,
            'bathrooms': 1,
            'stories': 1,
            'mainroad': 'yes',
            'guestroom': 'no',
            'basement': 'no',
            'hotwaterheating': 'no',
            'airconditioning': 'no',
            'parking': 1,
            'prefarea': 'no',
            'furnishingstatus': 'unfurnished'
        }
    ]
    
    return sample_houses


if __name__ == "__main__":
    # Example usage
    predictor = HousePricePredictor()
    
    # Load model (assuming it exists)
    try:
        predictor.load_model('models/best_model.pkl', 'models/preprocessor.pkl')
        
        # Make sample predictions
        sample_houses = create_sample_predictions()
        print("🏠 Sample House Price Predictions:")
        print("-" * 60)
        
        for i, house in enumerate(sample_houses, 1):
            try:
                price = predictor.predict_single(house)
                print(f"\nHouse {i}:")
                print(f" Area: {house['area']} sq ft")
                print(f" Bedrooms: {house['bedrooms']}")
                print(f" Bathrooms: {house['bathrooms']}")
                print(f" Stories: {house['stories']}")
                print(f" Furnishing: {house['furnishingstatus']}")
                print(f" Predicted Price: ${price:,.2f}")
                
                # Get confidence interval if available
                try:
                    confidence = predictor.get_prediction_confidence(house)
                    if confidence['std'] > 0:
                        print(f" Confidence Interval: {confidence['confidence_interval']}")
                        print(f" Standard Deviation: ${confidence['std']:,.0f}")
                except:
                    pass  # Skip confidence interval if not available
                    
            except Exception as e:
                print(f" Error predicting house {i}: {e}")
                
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run the main pipeline first to train and save models.")
        print("Run: python main.py")
