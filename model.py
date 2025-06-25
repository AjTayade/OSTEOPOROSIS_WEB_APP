import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_and_train():
    df = pd.read_csv('osteoporosis.csv')  # Ensure this file is in the project root

    df.drop(columns=['Id'], inplace=True)
    df.fillna('None', inplace=True)

    label_cols = df.select_dtypes(include='object').columns
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop('Osteoporosis', axis=1)
    y = df['Osteoporosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1_score": round(f1_score(y_test, y_pred), 3),
    }

    return model, scaler, X.columns.tolist(), metrics
