import streamlit as st
import streamlit.components.v1 as components

from joblib import load
import pickle

import numpy as np
import pandas as pd

with open("../pipelines/pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

model = load("../models/finetuned/stacking_classifier.joblib")

linkedin_profile_badge = """
<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
<div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="ameur-b-25a155247" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://dz.linkedin.com/in/ameur-b-25a155247?trk=profile-badge">Ameur B.</a></div>
"""

st.set_page_config(
    page_title="FraudAvert",
    page_icon="🪙"
)
 
tab1, tab2, tab3 = st.tabs(["AI","Monthly Hour Explanation", "Linkedin"])
with tab2:
    st.text("maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).")
with tab3:
    components.html(linkedin_profile_badge)

with tab1:
    st.title("FraudAvert")
    st.subheader("Real-Time Fraud Detection for Financial Transactions")

    st.warning("Empty values degrades system performance, try to fill all features.")
    monthly_hour,type,amount,sender_balance,receiver_balance = st.columns([1,1.5,1.5,1.5,1.5])
    
    with st.form(key="features"): 
        with monthly_hour:
            monthly_hour_val = st.number_input("Monthly Hour",min_value=0,max_value=744)
        with type:
            type_val = st.selectbox(label="Type",options=["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"])
        
        with amount:
            amount_val = st.number_input("Amount",min_value=0)

        with sender_balance:
            sender_balance_val = st.number_input("Sender Balance",min_value=0)

        with receiver_balance:
            receiver_balance_val = st.number_input("Receiver Balance",min_value=0)
        
        submit_button = st.form_submit_button(label="Predict")
    
        data = pd.DataFrame(np.array([monthly_hour_val,type_val,amount_val,sender_balance_val,receiver_balance_val]).reshape(1,-1))
        data.columns = ["MonthlyHour","Type","Amount","SenderBalance","ReceiverBalance"]
        for col in ["MonthlyHour","Amount","SenderBalance","ReceiverBalance"]:
            data[col] = float(data[col])
                                      
        if submit_button:
            data = pipeline.transform(data)
            st.text(model.predict(data))   