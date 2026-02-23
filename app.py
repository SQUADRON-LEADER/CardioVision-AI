import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import os

class InsurancePredictor:
    def __init__(self):
        self.model = None
        self.r2_score = None
        self.adjusted_r2_score = None
        self.model_path = "insurance_model.pkl"
        
    def load_and_train_model(self, data_path="insurance.csv"):
        """Load data and train the insurance prediction model"""
        try:
            # Load data
            insurance_data = pd.read_csv(data_path)
            print(f"Loaded {len(insurance_data)} records from {data_path}")
            
            # Prepare features and target
            x = insurance_data.drop(columns=["charges", "region"])
            y = insurance_data["charges"]
            
            # Encode categorical variables
            x["sex"] = x["sex"].map({"female": 0, "male": 1})
            x["smoker"] = x["smoker"].map({"no": 0, "yes": 1})
            
            # Split data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(x_train, y_train)
            
            # Calculate R² score
            y_pred = self.model.predict(x_test)
            self.r2_score = r2_score(y_test, y_pred)
            
            n = x_test.shape[0]
            p = x_test.shape[1]
            self.adjusted_r2_score = 1 - (1 - self.r2_score) * (n - 1) / (n - p - 1)
            
            print(f"Model trained successfully!")
            print(f"R² Score: {self.r2_score:.4f}")
            print(f"Adjusted R² Score: {self.adjusted_r2_score:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def save_model(self):
        """Save the trained model to disk"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'r2_score': self.r2_score,
                'adjusted_r2_score': self.adjusted_r2_score
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {self.model_path}")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self):
        """Load a pre-trained model from disk"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.r2_score = model_data['r2_score']
            self.adjusted_r2_score = model_data['adjusted_r2_score']
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"No saved model found at {self.model_path}")
            return False
    
    def predict(self, age, sex, bmi, children, smoker):
        """Make a prediction for insurance charges"""
        if self.model is None:
            print("Model not loaded. Please train or load a model first.")
            return None
        
        # Encode categorical variables
        sex_encoded = 0 if sex.lower() == "female" else 1
        smoker_encoded = 0 if smoker.lower() == "no" else 1
        
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex_encoded],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker_encoded]
        })
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        
        return {
            'predicted_charges': round(prediction, 2),
            'input_summary': {
                'age': age,
                'sex': sex,
                'bmi': bmi,
                'children': children,
                'smoker': smoker
            },
            'model_performance': {
                'r2_score': self.r2_score,
                'adjusted_r2_score': self.adjusted_r2_score
            }
        }
    
    def get_model_info(self):
        """Get information about the current model"""
        if self.model is None:
            return "No model loaded"
        
        return {
            'model_type': 'Linear Regression',
            'r2_score': self.r2_score,
            'adjusted_r2_score': self.adjusted_r2_score,
            'features': ['age', 'sex', 'bmi', 'children', 'smoker']
        }

def main():
    """Main function to demonstrate the backend functionality"""
    predictor = InsurancePredictor()
    
    # Try to load existing model, if not available, train new one
    if not predictor.load_model():
        print("Training new model...")
        if predictor.load_and_train_model():
            predictor.save_model()
        else:
            print("Failed to train model. Exiting.")
            return
    
    # Display model information
    print("\n" + "="*50)
    print("INSURANCE CHARGE PREDICTOR - BACKEND")
    print("="*50)
    model_info = predictor.get_model_info()
    print(f"Model Type: {model_info['model_type']}")
    print(f"R² Score: {model_info['r2_score']:.4f}")
    print(f"Adjusted R² Score: {model_info['adjusted_r2_score']:.4f}")
    print(f"Features: {', '.join(model_info['features'])}")
    
    # Example predictions
    print("\n" + "-"*30)
    print("EXAMPLE PREDICTIONS")
    print("-"*30)
    
    examples = [
        (30, "male", 25.0, 0, "no"),
        (45, "female", 30.5, 2, "yes"),
        (25, "female", 22.0, 1, "no")
    ]
    
    for i, (age, sex, bmi, children, smoker) in enumerate(examples, 1):
        result = predictor.predict(age, sex, bmi, children, smoker)
        if result:
            print(f"\nExample {i}:")
            print(f"  Input: Age={age}, Sex={sex}, BMI={bmi}, Children={children}, Smoker={smoker}")
            print(f"  Predicted Charges: ${result['predicted_charges']:,.2f}")

if __name__ == "__main__":
    main()
