import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8000/predict"

st.sidebar.title("Customer Behavior Analysis and Prediction System.")

with st.form("Data Input Form"):
    st.subheader("Features Input Form")
    ft_1 = st.slider("Age", min_value=0, max_value=90)
    ft_2 = st.text_input("Gender", value="Male").capitalize()
    ft_3 = st.slider("Tenure", 0.0, 1000.0, 0.0)
    ft_4 = st.text_input(label="Desktop", value="Mobile").capitalize()
    ft_5 = st.slider(label="CityTier", value=0)
    ft_6 = st.slider("WarehouseToHome", 0.0, 1000.0, 0.0)
    ft_7 = st.slider("HoursSpentOnApp", 0.0, 1000.0, 0.0)
    ft_8 = st.slider(label="NumberOfDevicesRegistered", value=0)
    ft_9 = st.slider(label="SatisfactionScore", value=0)
    ft_10 = st.slider(label="NumberOfAddress", value=0)
    ft_11 = st.slider(label="OrderCount", value=0)
    ft_12 = st.slider(label="DaySinceLastOrder", value=0)
    ft_13 = st.slider("CashbackAmount", 0.0, 1000.0, 0.0)

    submit = st.form_submit_button("Performing predictive analysis")
data = {
    "Age": ft_1, "Gender": ft_2, "Tenure": ft_3, "PreferredLoginDevice": ft_4,
    "CityTier": ft_5, "WarehouseToHome": ft_6, "HoursSpentOnApp": ft_7, "NumberOfDevicesRegistered": ft_8,
    "SatisfactionScore": ft_9, "NumberOfAddress": ft_10, "OrderCount": ft_11, "DaySinceLastOrder": ft_12,
    "CashbackAmount": ft_13,
}

df = pd.DataFrame([data])

placeholder = st.empty()

if submit:
    if (ft_2 in ["Male", "Female"]) and (ft_4 in ["Mobile", "Desktop"]):
        payload = {"samples": [
            {"Age": ft_1,
             "Gender": ft_2,
             "Tenure": ft_3,
             "PreferredLoginDevice": ft_4,
             "CityTier": ft_5,
             "WarehouseToHome": ft_6,
             "HoursSpentOnApp": ft_7,
             "NumberOfDevicesRegistered": ft_8,
             "SatisfactionScore": ft_9,
             "NumberOfAddress": ft_10,
             "OrderCount": ft_11,
             "DaySinceLastOrder": ft_12,
             "CashbackAmount": ft_13}
            ]}
    else:
        st.error("Please fill in the Gender or PreferredLoginDevice fields")

    with st.spinner("Sending request to the container"):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                list_val = result.get("prediction result")
                if isinstance(list_val, list):
                    extracted_value = list_val[0]
                    if extracted_value == 0:
                        final_val = "Loyal Customer 📈"

                        st.toast("Prediction completed successfully.")
                        st.info("prediction result")
                        st.metric(label="finall result", value=final_val)
                        st.sidebar.info(final_val)
                        st.sidebar.dataframe(df)
                    else:
                        final_val = "At risk of churn 📉"

                        st.toast("Prediction completed successfully.")
                        st.info("prediction result")
                        st.metric(label="finall result", value=final_val)
                        st.sidebar.info(final_val)
                        st.sidebar.dataframe(df)

            else:
                st.error(f"server error : {response.status_code}")
        except Exception as e:
            st.error("Unexpected error")