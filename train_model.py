import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def train_and_save():
    # 1. Load Data
    df = pd.read_csv('Churn_Modelling.csv')
    X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    y = df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define Features
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    categorical_features = ['Geography', 'Gender']

    # 3. Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 4. Pipeline with SMOTE
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 5. Fit and Save
    print("Training the pipeline...")
    model_pipeline.fit(X_train, y_train)
    
    joblib.dump(model_pipeline, 'churn_model_prod.pkl')
    joblib.dump((X_test, y_test, numeric_features, categorical_features), 'test_assets.pkl')
    print("✅ Model and Assets saved successfully.")

if __name__ == "__main__":
    train_and_save()