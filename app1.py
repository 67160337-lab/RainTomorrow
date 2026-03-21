import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Rain Predictor AU", layout="wide")
st.title("🌧️ เครื่องมือพยากรณ์โอกาสเกิดฝนในวันพรุ่งนี้ (Australia)")

# --- 1. โหลดข้อมูล (Target: RainTomorrow) ---
@st.cache_data
def load_data():
    df = pd.read_csv('weatherAUS.csv')
    # เลือก Features ที่ส่งผลต่อการเกิดฝน
    features_list = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm', 'Temp3pm', 'RainToday', 'RainTomorrow']
    df = df[features_list].dropna(subset=['RainTomorrow']) # ต้องตัดค่าว่างใน Target ออก
    
    # แปลง Target (Yes/No) เป็นตัวเลข (1/0) เพื่อให้โมเดลประมวลผลได้
    df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
    df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
    return df

df = load_data()

# --- 2. การเตรียม Pipeline (Classification) ---
num_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm', 'Temp3pm']
cat_features = ['Location']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. การเทรนโมเดลที่ดีที่สุด (XGBClassifier) ---
@st.cache_resource
def train_rain_model():
    # เปลี่ยนจาก XGBRegressor เป็น XGBClassifier
    main_pipe = Pipeline(steps=[('preprocessor', preprocessor), 
                                ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))])
    
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # ใช้ scoring='accuracy' หรือ 'f1' สำหรับงานจำแนกประเภท
    grid_search = GridSearchCV(main_pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

best_model = train_rain_model()

# --- 4. ส่วนการทำนายผล ---
st.header("🔮 ตรวจสอบโอกาสฝนตกวันพรุ่งนี้")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        u_loc = st.selectbox("เลือกสถานที่", sorted(df['Location'].unique()))
        u_min = st.number_input("อุณหภูมิต่ำสุดวันนี้ (°C)", value=15.0)
        u_max = st.number_input("อุณหภูมิสูงสุดวันนี้ (°C)", value=25.0)
        u_rtoday = st.selectbox("วันนี้ฝนตกหรือไม่?", ["No", "Yes"])
    with col2:
        u_rain = st.number_input("ปริมาณน้ำฝนวันนี้ (mm)", value=0.0)
        u_hum = st.slider("ความชื้นเวลา 15:00 (%)", 0, 100, 50)
        u_pres = st.number_input("ความกดอากาศ (hPa)", value=1015.0)
        u_temp3 = st.number_input("อุณหภูมิเวลา 15:00 (°C)", value=20.0)

if st.button("ทำนายผล"):
    # เตรียมข้อมูล Input (แปลง RainToday เป็น 0/1 ตามที่โมเดลเทรนมา)
    rtoday_val = 1 if u_rtoday == "Yes" else 0
    input_data = pd.DataFrame({
        'Location': [u_loc], 'MinTemp': [u_min], 'MaxTemp': [u_max], 
        'Rainfall': [u_rain], 'Humidity3pm': [u_hum], 'Pressure3pm': [u_pres],
        'Temp3pm': [u_temp3], 'RainToday': [rtoday_val]
    })
    
    prediction = best_model.predict(input_data)
    prob = best_model.predict_proba(input_data)[0][1] # ดึงค่าความน่าจะเป็นออกมาโชว์
    
    if prediction[0] == 1:
        st.error(f"🌧️ **คาดการณ์ว่า: พรุ่งนี้ฝนตก** (โอกาส {prob*100:.2f}%)")
    else:
        st.success(f"☀️ **คาดการณ์ว่า: พรุ่งนี้ฝนไม่ตก** (โอกาสฝนตกเพียง {(1-prob)*100:.2f}%)")