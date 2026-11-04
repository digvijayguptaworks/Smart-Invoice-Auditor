import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "freight_cost_predcition" / "models" / "predict_invoice_flag_model.pkl"
)

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]


def load_model(model_path=MODEL_PATH):
    """Load trained invoice flagging model."""
    return joblib.load(model_path)


def predict_invoice_flag(input_data):
    """Predict whether invoices need manual approval (1) or not (0)."""
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df["Predicted_Flag"] = model.predict(input_df[FEATURES]).astype(int)
    return input_df


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.0],
    }
    print(predict_invoice_flag(sample_data))