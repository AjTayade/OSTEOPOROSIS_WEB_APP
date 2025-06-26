from flask import Flask, render_template, request
from model import load_and_train
import numpy as np
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# --- Load Model and Preprocessors ---
# This is done once when the application starts.
try:
    model, scaler, label_encoders, feature_list, metrics, form_options = load_and_train()
except Exception as e:
    # If model loading fails, print a clear error and exit or handle gracefully.
    print(f"FATAL: Failed to load model on startup: {e}")
    # In a real app, you might want to exit or serve an error page.
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

            # Create a dictionary from the form data
            input_data_dict = {feature: request.form.get(feature) for feature in feature_list}
            
            # Convert the dictionary to a DataFrame, which is easier to work with
            input_df = pd.DataFrame([input_data_dict])
            
            # Process the input DataFrame
            # Apply label encoding to categorical features
            for col, le in label_encoders.items():
                # Get the value from the first row of the column
                form_value = input_df.at[0, col]
                
                # Check if the value is a known category
                if form_value not in le.classes_:
                    raise ValueError(f"Invalid value '{form_value}' for the field '{col}'.")
                
                # Transform the value using the loaded encoder
                input_df[col] = le.transform(input_df[col])

            # Ensure all data is numeric before scaling
            input_df = input_df.astype(float)

            # --- Make Prediction ---

            # Scale the processed input data
            input_scaled = scaler.transform(input_df)
            
            # Predict using the model
            pred_value = model.predict(input_scaled)[0]
            
            # Get the probability of having osteoporosis
            pred_proba = model.predict_proba(input_scaled)[0][1] # Probability of class '1'

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
            error_message = 'Invalid input. Please ensure all fields are filled correctly with valid numbers.'

    # --- Render the Page ---
    # Pass all necessary variables to the HTML template
    return render_template('index.html',
                           prediction=prediction_result,
                           metrics=metrics,
                           feature_list=feature_list,
                           form_options=form_options,
                           error=error_message)

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
