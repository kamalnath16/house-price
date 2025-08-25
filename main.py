"""
Main script to run the house price prediction pipeline
"""

import os
import numpy as np
import pandas as pd

from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.prediction import HousePricePredictor

def create_sample_data():
    """Create sample housing data for demonstration"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        'sqft': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'garage': np.random.randint(0, 4, n_samples),
        'location_score': np.random.uniform(0.3, 1.0, n_samples),
        'property_type': np.random.choice(['House', 'Apartment', 'Condo'], n_samples)
    }

    # Create realistic price based on features
    df = pd.DataFrame(data)
    df['price'] = (
        df['sqft'] * 150 +
        df['bedrooms'] * 10000 +
        df['bathrooms'] * 15000 +
        (50 - df['age']) * 2000 +
        df['garage'] * 8000 +
        df['location_score'] * 100000 +
        np.where(df['property_type'] == 'House', 50000,
                 np.where(df['property_type'] == 'Condo', 20000, 0)) +
        np.random.normal(0, 30000, n_samples)
    )

    # Ensure positive prices
    df['price'] = np.maximum(df['price'], 50000)

    return df

def main():
    """Main execution function"""
    print("🏠 House Price Prediction ML Project")
    print("=" * 40)

    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Step 1: Create/Load Data
    print("📊 Loading data...")
    data_path = 'data/raw/housing_data.csv'

    if not os.path.exists(data_path):
        print("Creating sample dataset...")
        df = create_sample_data()
        df.to_csv(data_path, index=False)
        print(f"Sample data saved to {data_path}")
    else:
        df = pd.read_csv(data_path)
        print(f"Data loaded from {data_path}")

    print(f"Dataset shape: {df.shape}")
    print(f"Target variable (price) range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")

    # Step 2: Data Preprocessing
    print("\n🔧 Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)

    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")

    # Step 3: Model Training
    print("\n🤖 Training models...")
    trainer = ModelTrainer()
    models = trainer.train_models(X_train, y_train)

    print(f"Trained {len(models)} models:")
    for name in models.keys():
        print(f" ✓ {name}")

    # Step 4: Model Evaluation
    print("\n📈 Evaluating models...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_models(models, X_test, y_test)

    print("\nModel Performance:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name}:")
        print(f" MAE: ${metrics['mae']:,.0f}")
        print(f" RMSE: ${metrics['rmse']:,.0f}")
        print(f" R² Score: {metrics['r2']:.4f}")
        print()

    # Step 5: Save Best Model
    best_model_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best_model = models[best_model_name]

    import joblib
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    print(f"🏆 Best model: {best_model_name}")
    print(f"Best model saved to: models/best_model.pkl")

    # Step 6: Example Prediction
    print("\n🔮 Making sample predictions...")
    predictor = HousePricePredictor()
    predictor.load_model('models/best_model.pkl', 'models/preprocessor.pkl')

    # Sample house for prediction
    sample_house = {
        'sqft': 2500,
        'bedrooms': 4,
        'bathrooms': 3,
        'age': 15,
        'garage': 2,
        'location_score': 0.8,
        'property_type': 'House'
    }

    predicted_price = predictor.predict_single(sample_house)

    print(f"\nSample House Features:")
    for key, value in sample_house.items():
        print(f" {key}: {value}")

    print(f"\n💰 Predicted Price: ${predicted_price:,.2f}")

    print("\n✅ Pipeline completed successfully!")
    print("📁 Check the 'models' directory for saved models")
    print("📊 Check the 'data' directory for processed datasets")

if __name__ == "__main__":
    main()
