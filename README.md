## 🌧️ Australia Rain Tomorrow Predictor
**โปรเจคพยากรณ์โอกาสเกิดฝนตกในวันถัดไปของประเทศออสเตรเลียด้วย Machine Learning**

แอปพลิเคชันนี้ถูกพัฒนาขึ้นเพื่อวิเคราะห์และพยากรณ์ว่า "พรุ่งนี้ฝนจะตกหรือไม่" โดยใช้ข้อมูลปัจจัยทางอุตุนิยมวิทยาเชิงลึก เช่น ความชื้นตอนบ่าย, ความกดอากาศ และความต่างของอุณหภูมิ เพื่อช่วยในการตัดสินใจและวางแผนล่วงหน้าได้อย่างแม่นยำ

## 🚀 เทคโนโลยีที่ใช้ (Tech Stack)
* **Language:** Python
* **Framework:** Streamlit
* **Machine Learning:** * **XGBoost:** โมเดลหลักที่เลือกใช้ (Champion Model) เนื่องจากจัดการข้อมูลที่ไม่สมดุลได้ดี
    * **Logistic Regression & Random Forest:** ใช้สำหรับเปรียบเทียบประสิทธิภาพ
* **Data Processing:** Scikit-learn (Pipeline, ColumnTransformer, SimpleImputer, Standard Scaler, OneHotEncoder)
* **Visualization:** Matplotlib, Seaborn (สำหรับการแสดง Feature Importance)

## 🛠️ วิธีการติดตั้งและใช้งาน
1. **Clone Repository:**
   ```bash
   git clone [https://github.com/67160337-lab/RainTomorrow.git](https://github.com/67160337-lab/RainTomorrow.git)
   cd RainTomorrow

2. **ติดตั้ง Library ที่จำเป็น:**
   ```bash
   pip install -r requirements.txt

3. **รันแอปพลิเคชัน:**
   ```bash
   streamlit run app1.py

## 🧠กระบวนการพัฒนา (Methodology)
* **Data Cleaning:** จัดการข้อมูลสูญหาย (Missing Values) ด้วยเทคนิคที่เหมาะสม และแปลงข้อมูลหมวดหมู่ด้วย OneHotEncoding

* **Feature Engineering:** สร้างตัวแปรใหม่เพื่อเพิ่มความแม่นยำ เช่น PressureDiff (ความต่างความกดอากาศ) และ TempDiff (ความต่างอุณหภูมิ)

* **Handling Imbalanced Data:** ใช้พารามิเตอร์ scale_pos_weight ใน XGBoost เพื่อแก้ปัญหาจำนวนวันที่ฝนไม่ตกมีมากกว่าวันที่ฝนตก

* **Optimization:** ใช้เทคนิค Threshold Tuning เพื่อหาจุดตัดสินใจที่เหมาะสมที่สุด เพื่อดันค่า F1-Score ให้สูงกว่าเกณฑ์มาตรฐาน

## 📊ผลลัพธ์ของโมเดล (Model Results)

**จากการเปรียบเทียบ 3 โมเดล พบว่า XGBoost ให้ประสิทธิภาพสูงสุดในแง่ของความสมดุล:**

* **Accuracy:** ~85.00% (ความแม่นยำในการทำนายภาพรวม)

* **F1-Score:** ~0.5900 (ประสิทธิภาพในการดึงสัญญาณวันฝนตกออกมาได้แม่นยำที่สุด)

* **Top Feature:** ปัจจัยที่มีผลต่อการทำนายมากที่สุดคือ Humidity3pm (ความชื้นตอนบ่าย)

## 💡 วิธีการใช้งาน (Usage)

**ระบุข้อมูลสภาพอากาศ:** กรอกค่าอุณหภูมิ, ความชื้น, ความกดอากาศ และปริมาณน้ำฝนปัจจุบัน

**วิเคราะห์ปัจจัย:** ระบบจะแสดงกราฟ Feature Importance เพื่อบอกว่าปัจจัยใดมีผลต่อการทำนายครั้งนั้นมากที่สุด

**ผลการพยากรณ์:** ระบบจะแสดงผลว่า "ฝนน่าจะตก" หรือ "ไม่น่าจะตก" พร้อมระบุค่าความเชื่อมั่น (Confidence Score) เป็นเปอร์เซ็นต์

## ⚠️ ข้อควรระวัง (Disclaimer)

**แบบจำลองนี้เป็นส่วนหนึ่งของการศึกษาในวิชา Data Science ผลการพยากรณ์เป็นการคำนวณเชิงสถิติ ไม่ควรใช้เพื่อการตัดสินใจที่ส่งผลกระทบต่อชีวิตและทรัพย์สินในสถานการณ์จริง**

## 👤 จัดทำโดย

**67160337 ธนพล แสงนวล**

**วิชา: Data Science**
