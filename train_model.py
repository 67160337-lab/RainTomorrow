import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# [หมวดที่ 1] นิยามปัญหาและ Dataset [cite: 13]
df = pd.read_csv('weatherAUS.csv')

# [หมวดที่ 2] การเตรียมข้อมูลและ EDA [cite: 19]
# ลบตัวแปรที่ทำให้เกิด Data Leakage (RISK_MM) และตัวแปรที่มีค่าว่างสูง [cite: 21]
cols_to_drop = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Date', 'Location', 'RISK_MM']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
df = df.dropna(subset=['RainTomorrow'])

# Feature Engineering: ใช้เฉพาะตัวแปรสำคัญเพื่อความง่ายใน App
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

# สร้าง Pipeline แยกตามประเภทข้อมูล [cite: 21]
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# [หมวดที่ 3] Model Development (ใช้ Time-based Split ตามหลัก Time Series) [cite: 25]
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# รวม Preprocessing และ Model เข้าใน Pipeline เดียว [cite: 21]
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])

# Train Model
model_pipeline.fit(X_train, y_train)

# ประเมินผลด้วย F1-Score (เพราะข้อมูล Imbalanced) [cite: 27, 28]
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# บันทึกโมเดลไว้ใช้ใน Streamlit [cite: 35]
joblib.dump(model_pipeline, 'rain_model.pkl')
print("Model saved as rain_model.pkl")