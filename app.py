from flask import Flask, render_template, request
from model import load_and_train
import numpy as np
import pandas as pd
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Load Model and Preprocessors ---
# This is done once when the application starts.
try:
    model, scaler, label_encoders, feature_list, metrics, form_options = load_and_train()
    print("Model and preprocessors loaded successfully!")
    # Debug: Print classes for the problematic encoder if it exists
    if 'Hormonal_Changes' in label_encoders:
        print(f"Hormonal_Changes LabelEncoder classes: {label_encoders['Hormonal_Changes'].classes_}")
    else:
        print("Hormonal_Changes not found in label_encoders after load_and_train. Check feature_list and categorical_cols in model.py.")

except Exception as e:
    # If model loading fails, print a clear error and exit or handle gracefully.
    print(f"FATAL: Failed to load model on startup: {e}")
    model = None # Set to None to handle in routes

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles both GET requests (displaying the form) and POST requests (making predictions).
    """
    if model is None:
        # If the model failed to load, show an error message.
        return "The prediction model could not be loaded. Please check the server logs.", 500

    prediction_result = None
    error_message = None

    if request.method == 'POST':
        try:
            # --- Collect and Preprocess User Input ---

            input_data_raw = {}
            for feature in feature_list:
                # Retrieve value from form. Use a default empty string if the field is not present.
                val = request.form.get(feature, '') 
                input_data_raw[feature] = val
                print(f"DEBUG: Form input for '{feature}': '{val}' (Type: {type(val)})") # Debug print

            # Convert the raw input data to a DataFrame
            # Create a Series first to ensure consistent dtype handling, then convert to DataFrame
            input_series = pd.Series(input_data_raw)
            input_df = pd.DataFrame([input_series])
            
            # Process the input DataFrame
            for col in feature_list: # Iterate through all expected features
                current_value = input_df.at[0, col]

                if col == 'Age':
                    # Age is numeric, convert it
                    if not current_value: # Handle empty string for age
                        raise ValueError("Age field cannot be empty.")
                    # Attempt conversion to numeric. 'coerce' will turn invalid parsing into NaN.
                    input_df[col] = pd.to_numeric(current_value, errors='coerce')
                    if pd.isna(input_df.at[0, col]): # Check if conversion failed
                        raise ValueError(f"Invalid numeric value for Age: '{current_value}'.")
                elif col in label_encoders:
                    # Categorical feature: apply label encoding
                    le = label_encoders[col]
                    
                    # IMPORTANT: If the value is an empty string, we need a default.
                    # This default MUST be one of the classes the LabelEncoder was fitted on.
                    if current_value == '':
                        # Choose a default for empty strings based on your data.
                        # For Hormonal_Changes, 'Normal' or 'Postmenopausal' are valid.
                        # 'Normal' is a common default if not specifically known.
                        # You MUST ensure this default is in le.classes_
                        if col == 'Hormonal_Changes':
                            default_val = 'Normal' # Or 'Postmenopausal', based on your domain
                        else:
                            # For other categorical columns, pick a suitable default from their classes_
                            # This assumes le.classes_ is not empty.
                            default_val = le.classes_[0] if len(le.classes_) > 0 else ''
                            print(f"DEBUG: Empty value for '{col}', defaulting to '{default_val}'")
                        current_value = default_val
                        input_df.at[0, col] = current_value # Update value in df

                    # Check if the value is a known category for the LabelEncoder
                    if current_value not in le.classes_:
                        # This is where your "Invalid value 'None'" error arises if current_value is 'None'
                        # and 'None' is not in le.classes_
                        raise ValueError(f"Invalid value '{current_value}' for the field '{col}'. "
                                         f"Expected one of: {', '.join(le.classes_)}")
                    
                    # Transform the value using the loaded encoder. le.transform expects an array-like input.
                    input_df[col] = le.transform([current_value])[0] 
                else:
                    # Fallback for any other unexpected column not in feature_list or label_encoders
                    # This case should ideally not be reached if feature_list is comprehensive
                    print(f"WARNING: Column '{col}' not found in label_encoders and not 'Age'. Attempting numeric conversion.")
                    input_df[col] = pd.to_numeric(current_value, errors='coerce')
                    if pd.isna(input_df.at[0, col]):
                        raise ValueError(f"Invalid numeric or unexpected value for field '{col}': '{current_value}'.")

            # Ensure the order of columns in input_df matches the order the model expects
            # This is CRITICAL for correct predictions.
            input_df = input_df[feature_list] 
            
            # --- Make Prediction ---

            # Scale the processed input data
            input_scaled = scaler.transform(input_df)
            
            # Predict using the model
            pred_value = model.predict(input_scaled)[0]
            
            # Get the probability of having osteoporosis (class '1')
            pred_proba = model.predict_proba(input_scaled)[0][1] 

            # Format the prediction for display
            risk = 'High Risk' if pred_value == 1 else 'Low Risk'
            confidence = f"{pred_proba * 100:.2f}%"
            prediction_result = f"{risk} of Osteoporosis (Confidence: {confidence})"

        except ValueError as ve:
            # Handle specific value errors, like invalid form inputs
            print(f"ValueError during prediction: {ve}")
            error_message = str(ve)
        except Exception as e:
            # Handle any other unexpected errors during prediction
            print(f"An unexpected error occurred during prediction: {e}")
            error_message = f'An unexpected error occurred: {e}. Please ensure all fields are filled correctly.'

    # --- Render the Page ---
    # Pass all necessary variables to the HTML template
    return render_template('index.html',
                           prediction=prediction_result,
                           metrics=metrics,
                           feature_list=feature_list,
                           form_options=form_options,
                           error=error_message)

if __name__ == '__main__':
    # Get the port from the environment variable, default to 5000 if not set
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
