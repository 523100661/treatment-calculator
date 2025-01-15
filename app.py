import streamlit as st
import joblib
import numpy as np

# 加载保存的模型和标准化器
model = joblib.load('final_logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit 应用标题
st.title("Disease Control Rate (DCR) Calculator")
st.write("Enter the following parameters to calculate the probability of disease control. A higher probability indicates a better prognosis.")

# 创建输入表单
diameter = st.number_input("Diameter (cm)", min_value=0.0, step=0.1, format="%.2f")
plt = st.number_input("PLT (10⁹/L)", min_value=0.0, step=0.1, format="%.2f")
stiffness_tumor_increase = st.radio(
    "Stiffness Tumor Increase",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)
wbc = st.number_input("WBC (10⁹/L)", min_value=0.0, step=0.1, format="%.2f")

# 提交按钮
if st.button("Calculate"):
    try:
        # 构造输入数组
        features = np.array([[diameter, plt, stiffness_tumor_increase, wbc]])
        
        # 标准化输入特征（使用保存的标准化器）
        features_scaled = scaler.transform(features)

        # 进行概率预测
        probability = model.predict_proba(features_scaled)[0, 1]  # 获取阳性概率

        # 显示预测结果
        st.success(f"Predicted Disease Control Rate (DCR): {round(probability * 100, 2)}%")
    except Exception as e:
        st.error(f"An error occurred: {e}")
