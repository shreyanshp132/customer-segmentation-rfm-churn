import streamlit as st
import pandas as pd
import joblib

st.title("Smart Churn Analysis Dashboard")
model = joblib.load('churn_model.pkl')
frequency = st.number_input("Enter Frequency (number of orders)")
monetary = st.number_input("Enter Monetary (total amount spent)")
st.markdown("""
**Cluster Guide:**
- 0 = Lost/Inactive customers
- 1 = Potential Loyalists  
- 2 = Champions (highest value)
- 3 = Loyal Customers
- 4 = High-Value Customers
""")
cluster = st.selectbox("Select Cluster", [0,1,2,3,4])
if st.button("Predict"):
    new_customer = pd.DataFrame({'Frequency': [frequency], 'Monetary': [monetary], 'Cluster': [cluster]})
    prediction = model.predict(new_customer)
    
    if prediction[0] == 1:
        st.error("⚠️ High Churn Risk")
    else:
        st.success("✅ Low Churn Risk")