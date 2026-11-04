from modeling_evaluation import train_random_forest, evaluate_classifier
import joblib
from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features
)
import os 

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    # Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, 'invoice_flagging/models/scaler.pkl'
    )

    # Train and evaluate models
    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save best model
    os.makedirs('invoice_flagging/models', exist_ok=True)

    joblib.dump(
    grid_search.best_estimator_,
    'invoice_flagging/models/predict_flag_invoice.pkl'
    )


if __name__ == "__main__":
    main()