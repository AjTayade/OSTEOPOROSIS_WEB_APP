import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_and_train():
    """
    Loads the osteoporosis dataset, preprocesses the data, trains a logistic
    regression model, and returns all necessary components for prediction.
    """
    # Load the dataset
    try:
        df = pd.read_csv('osteoporosis.csv')
    except FileNotFoundError:
        # Provide a helpful error message if the data file is not found
        raise FileNotFoundError("Error: 'osteoporosis.csv' not found. Make sure the CSV file is in the same directory as your application.")

    # --- Data Preprocessing ---

    # Drop the 'Id' column as it's not needed for modeling
    if 'Id' in df.columns:
        df.drop(columns=['Id'], inplace=True)

    # Fill any missing values with the string 'None'
    # This ensures that missing data is treated as a distinct category
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna('None')

    # Identify categorical columns and apply label encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Exclude the target variable from the categorical columns list if it's there
    target_col = 'Osteoporosis'
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    # Encode the target variable separately
    if df[target_col].dtype == 'object':
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col])

    # Store label encoders for each categorical feature
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Separate features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # --- Model Training ---

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # --- Performance Evaluation ---

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1-Score": round(f1_score(y_test, y_pred), 3),
    }
    
    # The columns in the order the model expects them
    feature_list = X.columns.tolist()
    
    # Create a dictionary of options for the dropdowns in the frontend
    # This is derived from the classes_ of the fitted label encoders
    form_options = {col: le.classes_.tolist() for col, le in label_encoders.items()}

    return model, scaler, label_encoders, feature_list, metrics, form_options
