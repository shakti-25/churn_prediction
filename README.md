# 📊 Telco Customer Churn Prediction

A Flask-based Machine Learning web application that predicts whether a telecom customer is likely to churn using an XGBoost classification model. The application performs data preprocessing, feature engineering, and displays both churn probability and prediction results through an interactive web interface.

---

## 🚀 Features

- Predict customer churn in real time
- User-friendly Flask web interface
- Data preprocessing before prediction
- One-hot encoding and feature alignment
- Displays churn probability
- XGBoost machine learning model
- Responsive and simple UI

---

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- XGBoost
- Joblib
- HTML
- CSS

---

## 📁 Project Structure

```
Telco-Churn-Prediction/
│
├── app.py
├── models/
│   └── churn_xgb.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── requirements.txt
└── README.md
```

---

## 📊 Machine Learning Workflow

1. Load the trained XGBoost model.
2. Collect customer information through the web form.
3. Preprocess the input data.
4. Convert categorical features using one-hot encoding.
5. Align features with the training dataset.
6. Predict customer churn probability.
7. Display the prediction result.

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Telco-Churn-Prediction.git
```

### Navigate to the project

```bash
cd Telco-Churn-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📋 Input Features

- Gender
- Partner
- Dependents
- Phone Service
- Paperless Billing
- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Payment Method

---

## 📈 Output

The application displays:

- Customer Churn Probability
- Customer Will Churn / Customer Will Stay

---

## 📸 Screenshot

Add screenshots of your application here.

Example:

```
screenshots/home.png
screenshots/result.png
```

---

## 🎯 Future Improvements

- User authentication
- Database integration
- Interactive dashboard
- Model performance visualization
- Cloud deployment (Render, Railway, or AWS)
- REST API support

---

## 👨‍💻 Author

**Shakti Thevar**


---



