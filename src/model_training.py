"""
Model training module for house price prediction
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

class ModelTrainer:
    """Handles training of multiple ML models for house price prediction"""

    def __init__(self):
        self.models = {}
        self.best_params = {}

    def train_models(self, X_train, y_train, use_grid_search=True):
        """
        Train multiple models on the training data

        Args:
            X_train: Training features
            y_train: Training targets
            use_grid_search: Whether to use hyperparameter tuning

        Returns:
            Dictionary of trained models
        """
        print("Training Linear Regression...")
        self.models['Linear Regression'] = self._train_linear_regression(X_train, y_train)

        print("Training Ridge Regression...")
        self.models['Ridge Regression'] = self._train_ridge_regression(X_train, y_train, use_grid_search)

        print("Training Lasso Regression...")
        self.models['Lasso Regression'] = self._train_lasso_regression(X_train, y_train, use_grid_search)

        print("Training Random Forest...")
        self.models['Random Forest'] = self._train_random_forest(X_train, y_train, use_grid_search)

        print("Training XGBoost...")
        self.models['XGBoost'] = self._train_xgboost(X_train, y_train, use_grid_search)

        return self.models

    def _train_linear_regression(self, X_train, y_train):
        """Train a basic linear regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def _train_ridge_regression(self, X_train, y_train, use_grid_search=True):
        """Train a Ridge regression model with optional hyperparameter tuning"""
        if use_grid_search:
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
            model = Ridge()
            grid_search = GridSearchCV(
                model, param_grid, cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.best_params['Ridge'] = grid_search.best_params_
            return grid_search.best_estimator_
        else:
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            return model

    def _train_lasso_regression(self, X_train, y_train, use_grid_search=True):
        """Train a Lasso regression model with optional hyperparameter tuning"""
        if use_grid_search:
            param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
            model = Lasso(max_iter=2000)
            grid_search = GridSearchCV(
                model, param_grid, cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.best_params['Lasso'] = grid_search.best_params_
            return grid_search.best_estimator_
        else:
            model = Lasso(alpha=1.0, max_iter=2000)
            model.fit(X_train, y_train)
            return model

    def _train_random_forest(self, X_train, y_train, use_grid_search=True):
        """Train a Random Forest model with optional hyperparameter tuning"""
        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            model = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=3,  # Reduced CV folds for faster training
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.best_params['Random Forest'] = grid_search.best_params_
            return grid_search.best_estimator_
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train, y_train)
            return model

    def _train_xgboost(self, X_train, y_train, use_grid_search=True):
        """Train an XGBoost model with optional hyperparameter tuning"""
        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            model = xgb.XGBRegressor(random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=3,  # Reduced CV folds for faster training
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.best_params['XGBoost'] = grid_search.best_params_
            return grid_search.best_estimator_
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)
            return model

    def get_feature_importance(self, model_name):
        """Get feature importance from tree-based models"""
        if model_name in ['Random Forest', 'XGBoost']:
            model = self.models.get(model_name)
            if model and hasattr(model, 'feature_importances_'):
                return model.feature_importances_
        elif model_name in ['Ridge Regression', 'Lasso Regression', 'Linear Regression']:
            model = self.models.get(model_name)
            if model and hasattr(model, 'coef_'):
                return np.abs(model.coef_)
        return None

    def cross_validate_model(self, model_name, X, y, cv=5):
        """Perform cross-validation on a specific model"""
        if model_name in self.models:
            model = self.models[model_name]
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring='neg_mean_squared_error'
            )
            return {
                'mean_rmse': np.sqrt(-scores.mean()),
                'std_rmse': np.sqrt(scores.std()),
                'scores': -scores
            }
        return None
