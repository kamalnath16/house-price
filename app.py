from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import logging
from src.prediction import HousePricePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Production configuration
app.config['DEBUG'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Global variables for predictor
predictor = None
model_loaded = False

def load_model():
    """Load the trained model and preprocessor"""
    global predictor, model_loaded
    predictor = HousePricePredictor()
    try:
        model_path = 'models/best_model.pkl'
        preprocessor_path = 'models/preprocessor.pkl'
        
        # Check if model files exist
        if os.path.exists(model_path):
            predictor.load_model(model_path, preprocessor_path)
            logger.info("✅ Model loaded successfully")
            model_loaded = True
            return True
        else:
            logger.warning("⚠️ Model files not found - app will run without predictions")
            model_loaded = False
            return False
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        model_loaded = False
        return False

# Load model on startup (replaces @app.before_first_request)
with app.app_context():
    load_model()

@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle single house price prediction"""
    if not model_loaded or predictor is None or not predictor.is_loaded:
        error_msg = "Prediction service temporarily unavailable. Please try again later."
        logger.error(error_msg)
        return render_template('error.html', error=error_msg)
    
    try:
        # Extract form data
        house_features = {
            'area': float(request.form['area']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'stories': int(request.form['stories']),
            'mainroad': request.form['mainroad'],
            'guestroom': request.form['guestroom'],
            'basement': request.form['basement'],
            'hotwaterheating': request.form['hotwaterheating'],
            'airconditioning': request.form['airconditioning'],
            'parking': int(request.form['parking']),
            'prefarea': request.form['prefarea'],
            'furnishingstatus': request.form['furnishingstatus']
        }
        
        # Validate input
        if house_features['area'] < 500 or house_features['area'] > 20000:
            raise ValueError("Area must be between 500 and 20,000 sq ft")
        
        if house_features['bedrooms'] < 1 or house_features['bedrooms'] > 10:
            raise ValueError("Bedrooms must be between 1 and 10")
        
        if house_features['bathrooms'] < 1 or house_features['bathrooms'] > 10:
            raise ValueError("Bathrooms must be between 1 and 10")
        
        # Make prediction
        predicted_price = predictor.predict_single(house_features)
        
        logger.info(f"Prediction made: ${predicted_price:,.2f} for area: {house_features['area']} sq ft")
        
        return render_template('result.html', 
                             prediction=predicted_price,
                             features=house_features)
    
    except ValueError as ve:
        error_msg = f"Invalid input: {str(ve)}"
        logger.warning(error_msg)
        return render_template('error.html', error=error_msg)
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(error_msg)
        return render_template('error.html', error=error_msg)

@app.route('/batch_predict')
def batch_predict_page():
    """Batch prediction page"""
    if not model_loaded or predictor is None or not predictor.is_loaded:
        return render_template('error.html', 
                             error="Prediction service temporarily unavailable.")
    return render_template('batch_predict.html')

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions from CSV file"""
    if not model_loaded or predictor is None or not predictor.is_loaded:
        return render_template('error.html', 
                             error="Prediction service temporarily unavailable.")
    
    try:
        if 'file' not in request.files:
            return render_template('error.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', error="No file selected")
        
        if file and file.filename.endswith('.csv'):
            # Create temp directory if it doesn't exist
            os.makedirs('temp', exist_ok=True)
            
            # Save uploaded file temporarily
            filepath = os.path.join('temp', file.filename)
            file.save(filepath)
            
            try:
                # Make predictions
                results_df = predictor.predict_from_csv(filepath)
                
                # Clean up temp file
                os.remove(filepath)
                
                return render_template('batch_result.html', 
                                     results=results_df.to_html(classes='table table-striped table-hover', 
                                                               index=False))
            except Exception as e:
                # Clean up temp file if error occurs
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise e
        else:
            return render_template('error.html', 
                                 error="Please upload a CSV file")
            
    except Exception as e:
        error_msg = f"Batch prediction error: {str(e)}"
        logger.error(error_msg)
        return render_template('error.html', error=error_msg)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single predictions"""
    if not model_loaded or predictor is None or not predictor.is_loaded:
        return jsonify({
            'error': 'Prediction service unavailable', 
            'status': 'error'
        }), 503
    
    try:
        data = request.json
        if not data:
            return jsonify({
                'error': 'No data provided', 
                'status': 'error'
            }), 400
        
        # Validate required fields
        required_fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                          'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
                          'parking', 'prefarea', 'furnishingstatus']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'status': 'error'
            }), 400
        
        predicted_price = predictor.predict_single(data)
        
        return jsonify({
            'prediction': predicted_price,
            'formatted_prediction': f"${predicted_price:,.2f}",
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({
            'error': str(e), 
            'status': 'error'
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch predictions"""
    if not model_loaded or predictor is None or not predictor.is_loaded:
        return jsonify({
            'error': 'Prediction service unavailable', 
            'status': 'error'
        }), 503
    
    try:
        data = request.json
        if not data or 'houses' not in data:
            return jsonify({
                'error': 'No houses data provided', 
                'status': 'error'
            }), 400
        
        houses_list = data['houses']
        predictions = predictor.predict_batch(houses_list)
        
        results = []
        for i, (house, prediction) in enumerate(zip(houses_list, predictions)):
            results.append({
                'house_id': i + 1,
                'features': house,
                'prediction': prediction,
                'formatted_prediction': f"${prediction:,.2f}"
            })
        
        return jsonify({
            'results': results,
            'total_houses': len(results),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"API batch prediction error: {str(e)}")
        return jsonify({
            'error': str(e), 
            'status': 'error'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'version': '1.0.0',
        'service': 'house-price-predictor'
    })

@app.route('/model_info')
def model_info():
    """Get model information"""
    if not model_loaded or predictor is None or not predictor.is_loaded:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 503
    
    try:
        # Get model type
        model_type = type(predictor.model).__name__
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(predictor.model, 'feature_importances_'):
            feature_importance = predictor.model.feature_importances_.tolist()
        
        return jsonify({
            'model_type': model_type,
            'model_loaded': True,
            'has_feature_importance': feature_importance is not None,
            'feature_count': len(feature_importance) if feature_importance else 0,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('error.html', 
                          error="Page not found. Please check the URL."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return render_template('error.html', 
                          error="Internal server error. Please try again later."), 500

@app.errorhandler(503)
def service_unavailable_error(error):
    """Handle 503 errors"""
    return render_template('error.html', 
                          error="Service temporarily unavailable. Please try again later."), 503

if __name__ == '__main__':
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get('PORT', 10000))
    
    # For development, you can set debug=True
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info("🏠 House Price Predictor Starting...")
    logger.info(f"🌐 Port: {port}")
    logger.info(f"🤖 Model loaded: {model_loaded}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    
    app.run(
        host='0.0.0.0', 
        port=port,
        debug=debug_mode,
        threaded=True
    )
