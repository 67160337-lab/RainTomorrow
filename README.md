## 🌧️ Australia Rain Tomorrow Predictor
**โปรเจคพยากรณ์โอกาสเกิดฝนตกในวันถัดไปของประเทศออสเตรเลียด้วย Machine Learning**

แอปพลิเคชันนี้ถูกพัฒนาขึ้นเพื่อวิเคราะห์และพยากรณ์ว่า "พรุ่งนี้ฝนจะตกหรือไม่" โดยใช้ข้อมูลปัจจัยทางอุตุนิยมวิทยาเชิงลึก เช่น ความชื้นตอนบ่าย, ความกดอากาศ และความต่างของอุณหภูมิ เพื่อช่วยในการตัดสินใจและวางแผนล่วงหน้าได้อย่างแม่นยำ

## 🚀 เทคโนโลยีที่ใช้ (Tech Stack)
* **Language:** Python
* **Framework:** Streamlit
* **Machine Learning:** * **XGBoost:** โมเดลหลักที่เลือกใช้ (Champion Model) เนื่องจากจัดการข้อมูลที่ไม่สมดุลได้ดี
    * **Logistic Regression & Random Forest:** ใช้สำหรับเปรียบเทียบประสิทธิภาพ
* **Data Processing:** Scikit-learn (Pipeline, ColumnTransformer, Iterative Imputer, Standard Scaler, OneHotEncoder)
* **Visualization:** Matplotlib, Seaborn (สำหรับการแสดง Feature Importance)

## 🛠️ วิธีการติดตั้งและใช้งาน
1. **Clone Repository:**
   ```bash
   git clone [https://github.com/67160337-lab/Rain_Prediction_Australia.git](https://github.com/67160337-lab/Rain_Prediction_Australia.git)
   cd Rain_Prediction_Australia
