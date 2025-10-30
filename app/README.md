# Telco Customer Churn Prediction (Machine Learning Project)
---
title: "Telco Customer Churn Prediction"
description: "A machine learning project to predict telecom customer churn using Python, scikit-learn, and Streamlit."
author: "Shibu Jaiswal"
date: "2025-10-30"
version: "1.0.0"
license: "MIT"
tags: 
  - Data Science
  - Machine Learning
  - Streamlit
  - Python
  - RandomForest
  - Customer Churn
  - Predictive Analytics
repository: "https://github.com/shibujaiswal-cmyk/telco-customer-churn"
---


A complete data science project that predicts **customer churn** using a trained RandomForest model.  
It demonstrates the full end-to-end **data science workflow** — from data preparation and model training to deployment-ready app design in **Streamlit**.

---

## 🧠 Project Overview
This project predicts customer churn for a telecom company using a machine learning model trained on 30 customer-related features.  
It demonstrates the full end-to-end **data science workflow** — from data preparation and model training to deployment-ready app design in **Streamlit**.

The project is modular, cleanly structured, and production-friendly.  
It’s designed so that the same app can easily switch from dummy data to real datasets like the **Telco Customer Churn Dataset**.

---

## ⚙️ Tech Stack
- **Python 3.11**
- **Streamlit** – for interactive web UI  
- **scikit-learn** – for ML model (RandomForest)  
- **pandas, NumPy** – for data processing  
- **joblib** – for model serialization  

---

## 🧩 Key Features
✅ Built a dummy RandomForestClassifier model with 30 engineered features.  
✅ Used `StandardScaler` for feature normalization.  
✅ Modular folder structure (`app/`, `model/`, `src/`).  
✅ Error-handling for missing model artifacts.  
✅ Ready-to-deploy on Streamlit Cloud (one-click setup).  
✅ Highly customizable for real-world datasets.  

---

## 📁 Project Structure

---

## 🚀 How to Run Locally
```bash
# 1️⃣ Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Generate dummy model (optional)
python src/create_dummy_30f_model.py

# 4️⃣ Run the app
streamlit run app/app.py
