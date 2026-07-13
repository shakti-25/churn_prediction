from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("models/churn_xgb.pkl")


# ---------------- Preprocessing ---------------- #
def preprocess_input(df):

    binary_cols = [
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'PaperlessBilling'
    ]

    df[binary_cols] = df[binary_cols].replace({
        'Yes': 1,
        'No': 0,
        'Male': 1,
        'Female': 0
    })

    multi_category_cols = [
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaymentMethod'
    ]

    df = pd.get_dummies(df,
                        columns=multi_category_cols,
                        drop_first=True)

    if 'MultipleLines_No phone service' in df.columns:
        df['No phone service'] = df['MultipleLines_No phone service'].astype(int)
        df.drop(columns=['MultipleLines_No phone service'], inplace=True)

    if 'DeviceProtection_No internet service' in df.columns:

        df['No_internet_service'] = (
            df['DeviceProtection_No internet service'] |
            df['OnlineBackup_No internet service'] |
            df['OnlineSecurity_No internet service'] |
            df['StreamingTV_No internet service'] |
            df['StreamingMovies_No internet service'] |
            df['TechSupport_No internet service']
        ).astype(int)

        drop_cols = [c for c in df.columns if "No internet service" in c]
        df.drop(columns=drop_cols, inplace=True)

    df = df.fillna(df.median(numeric_only=True))

    return df


# ---------------- Home ---------------- #
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        input_dict = {
            "gender": [request.form["gender"]],
            "Partner": [request.form["partner"]],
            "Dependents": [request.form["dependents"]],
            "PhoneService": [request.form["phoneservice"]],
            "PaperlessBilling": [request.form["paperless"]],
            "tenure": [int(request.form["tenure"])],
            "MonthlyCharges": [float(request.form["monthly_charges"])],
            "TotalCharges": [float(request.form["total_charges"])],
            "Contract": [request.form["contract"]],
            "InternetService": [request.form["internet"]],
            "PaymentMethod": [request.form["payment"]],
            "MultipleLines": ["No"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"]
        }

        raw_df = pd.DataFrame(input_dict)

        processed = preprocess_input(raw_df)

        # Align columns with training
        for col in model.get_booster().feature_names:
            if col not in processed.columns:
                processed[col] = 0

        processed = processed[model.get_booster().feature_names]

        probability = model.predict_proba(processed)[0][1]
        pred = int(probability >= 0.5)

        prediction = "Customer Will Churn" if pred == 1 else "Customer Will Stay"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)