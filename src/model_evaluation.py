"""
Model evaluation module for house price prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

class ModelEvaluator:
    """Handles evaluation and visualization of ML models"""

    def __init__(self):
        self.results = {}
        plt.style.use('default')

    def evaluate_models(self, models, X_test, y_test):
        """
        Evaluate multiple models on test data

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics for each model
        """
        results = {}
        for name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }

        self.results = results
        self._create_comparison_plots(y_test)
        return results

    def _create_comparison_plots(self, y_test):
        """Create comparison plots for all models"""
        n_models = len(self.results)
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison', fontsize=16)

        # 1. Model Performance Comparison: MAE and RMSE
        metrics_df = pd.DataFrame({
            name: [results['mae'], results['rmse'], results['r2']]
            for name, results in self.results.items()
        }, index=['MAE', 'RMSE', 'R²'])

        metrics_df.loc[['MAE', 'RMSE']].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('MAE and RMSE Comparison')
        axes[0, 0].set_ylabel('Error')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. R² Score Comparison
        r2_scores = [results['r2'] for results in self.results.values()]
        model_names = list(self.results.keys())
        bars = axes[0, 1].bar(model_names, r2_scores, color='skyblue', alpha=0.7)

        axes[0, 1].set_title('R² Score Comparison')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')

        # 3. Residual Analysis (for best model)
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['r2'])
        best_predictions = self.results[best_model_name]['predictions']
        residuals = y_test - best_predictions

        axes[1, 0].scatter(best_predictions, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title(f'Residual Plot - {best_model_name}')

        # 4. Actual vs Predicted (for best model)
        min_val = min(y_test.min(), best_predictions.min())
        max_val = max(y_test.max(), best_predictions.max())

        axes[1, 1].scatter(y_test, best_predictions, alpha=0.6)
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title(f'Actual vs Predicted - {best_model_name}')

        plt.tight_layout()
        # Save the plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Model comparison plots saved to 'plots/model_comparison.png'")

    def create_feature_importance_plot(self, models, feature_names):
        """Create feature importance plots for tree-based models"""
        tree_models = ['Random Forest', 'XGBoost']
        available_models = [name for name in tree_models if name in models]

        if not available_models:
            print("No tree-based models available for feature importance analysis")
            return

        n_models = len(available_models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
        if n_models == 1:
            axes = [axes]

        for i, model_name in enumerate(available_models):
            model = models[model_name]
            importances = model.feature_importances_

            # Create a DataFrame for better visualization
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)

            # Plot top 15 features
            top_features = feature_importance_df.tail(15)

            axes[i].barh(top_features['feature'], top_features['importance'])
            axes[i].set_title(f'Feature Importance - {model_name}')
            axes[i].set_xlabel('Importance')

        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Feature importance plots saved to 'plots/feature_importance.png'")

    def generate_evaluation_report(self, y_test, save_to_file=True):
        """Generate a comprehensive evaluation report"""
        if not self.results:
            print("No evaluation results available. Run evaluate_models first.")
            return

        report = []
        report.append("=" * 60)
        report.append("HOUSE PRICE PREDICTION - MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")

        # Dataset summary
        report.append(f"Test Set Size: {len(y_test)} samples")
        report.append(f"Price Range: ${y_test.min():,.2f} - ${y_test.max():,.2f}")
        report.append(f"Mean Price: ${y_test.mean():,.2f}")
        report.append(f"Price Std: ${y_test.std():,.2f}")
        report.append("")

        # Model performance comparison
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("-" * 40)

        # Create a comparison table
        comparison_data = []

        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'MAE ($)': f"{results['mae']:,.0f}",
                'RMSE ($)': f"{results['rmse']:,.0f}",
                'R² Score': f"{results['r2']:.4f}",
                'MAPE (%)': f"{self._calculate_mape(y_test, results['predictions']):.2f}"
            })

        comparison_df = pd.DataFrame(comparison_data)
        report.append(comparison_df.to_string(index=False))
        report.append("")

        # Best model analysis
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['r2'])
        best_results = self.results[best_model_name]

        report.append("BEST MODEL ANALYSIS")
        report.append("-" * 25)
        report.append(f"Best Model: {best_model_name}")
        report.append(f"R² Score: {best_results['r2']:.4f}")
        report.append(f"Mean Absolute Error: ${best_results['mae']:,.0f}")
        report.append(f"Root Mean Squared Error: ${best_results['rmse']:,.0f}")
        report.append("")

        # Residual analysis
        residuals = y_test - best_results['predictions']
        report.append("RESIDUAL ANALYSIS")
        report.append("-" * 18)
        report.append(f"Mean Residual: ${residuals.mean():,.2f}")
        report.append(f"Std Residual: ${residuals.std():,.2f}")
        report.append(f"Max Absolute Residual: ${abs(residuals).max():,.2f}")
        report.append("")

        # Model ranking
        report.append("MODEL RANKING (by R² Score)")
        report.append("-" * 30)
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['r2'], reverse=True)
        for i, (name, results) in enumerate(sorted_models, 1):
            report.append(f"{i}. {name}: {results['r2']:.4f}")

        report_text = "\n".join(report)

        if save_to_file:
            os.makedirs('reports', exist_ok=True)
            with open('reports/evaluation_report.txt', 'w') as f:
                f.write(report_text)
            print("📄 Evaluation report saved to 'reports/evaluation_report.txt'")

        print("\n" + report_text)
        return report_text

    def _calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def plot_prediction_errors(self, y_test, model_name=None):
        """Plot prediction errors distribution"""
        if not self.results:
            print("No evaluation results available. Run evaluate_models first.")
            return

        # Use best model if none specified
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda k: self.results[k]['r2'])

        if model_name not in self.results:
            print(f"Model {model_name} not found in results")
            return

        predictions = self.results[model_name]['predictions']
        errors = predictions - y_test
        percentage_errors = (errors / y_test) * 100

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Absolute errors
        axes[0].hist(np.abs(errors), bins=30, alpha=0.7, color='skyblue')
        axes[0].set_xlabel('Absolute Error ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Absolute Errors')
        axes[0].axvline(np.mean(np.abs(errors)), color='red', linestyle='--',
                        label=f'Mean: ${np.mean(np.abs(errors)):,.0f}')
        axes[0].legend()

        # Percentage errors
        axes[1].hist(percentage_errors, bins=30, alpha=0.7, color='lightgreen')
        axes[1].set_xlabel('Percentage Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Percentage Errors')
        axes[1].axvline(np.mean(percentage_errors), color='red', linestyle='--',
                        label=f'Mean: {np.mean(percentage_errors):.1f}%')
        axes[1].legend()

        # Q-Q plot for residuals
        from scipy import stats
        residuals = y_test - predictions
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot of Residuals')

        plt.suptitle(f'Error Analysis - {model_name}')
        plt.tight_layout()

        plt.savefig('plots/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Error analysis plots saved to 'plots/error_analysis.png'")
