import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def run_analytics(clv=2000, retention_cost=200, success_rate=0.5):
    # 1. Load the saved assets from train_model.py
    try:
        model = joblib.load('churn_model_prod.pkl')
        X_test, y_test, num_feats, cat_feats = joblib.load('test_assets.pkl')
    except FileNotFoundError:
        print("Error: .pkl files not found. Please run train_model.py first.")
        return

    # 2. Basic Performance Metrics
    y_pred = model.predict(X_test)
    print("\n" + "="*30)
    print("MODEL PERFORMANCE REPORT")
    print("="*30)
    print(classification_report(y_test, y_pred))

    # 3. Financial Impact Logic
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # Logic: (True Positives * Success Rate * Value) - (Total Offers Sent * Cost)
    net_profit = (tp * success_rate * clv) - ((tp + fp) * retention_cost)
    
    print("--- BUSINESS ROI ---")
    print(f"Projected Net Profit: ${net_profit:,.2f}")
    print("="*30 + "\n")

    # 4. SHAP Explainability
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    # Transform raw X_test into the version the model actually used
    X_test_proc = preprocessor.transform(X_test)
    
    # Reconstruct Feature Names (Numeric + One-Hot Categories)
    cat_names = preprocessor.transformers_[1][1].get_feature_names_out(cat_feats)
    all_feature_names = num_feats + list(cat_names)

    # Initialize Explainer
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_proc)

    # Handle the SHAP value shape for Random Forest
    
    if isinstance(shap_values, list):
        final_shap = shap_values[1]
    else:
        # For newer versions of SHAP/sklearn, it might return a 3D array
        final_shap = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

    print("Generating Standard Summary Plot...")
    
    # Create the figure explicitly to control size
    plt.figure(figsize=(10, 8))

    # Trigger the standard Beeswarm/Dot plot
    shap.summary_plot(
        final_shap, 
        X_test_proc, 
        feature_names=all_feature_names, 
        max_display=len(all_feature_names), # Forces all columns to show
        plot_type="dot", 
        show=False
    )
    
    plt.title(f"Global Churn Drivers (Net ROI: ${net_profit:,.0f})", fontsize=14)
    plt.tight_layout() # Ensures labels don't overlap
    plt.show()

if __name__ == "__main__":
    # You can tweak these inputs for your interview demo
    run_analytics(clv=2000, retention_cost=200, success_rate=0.5)
