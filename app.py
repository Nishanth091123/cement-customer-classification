import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Cement Customer Classification",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Cement Customer Classification")
st.write("AI-based Cement Customer Classification using Machine Learning")

# Load dataset
dataset = pd.read_csv("cement_customer_classification_1000.csv")

# Load final model
with open("cement_customer_final_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load feature columns
with open("cement_customer_feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

st.header("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    customer_type = st.selectbox(
        "Customer Type",
        dataset["Customer_Type"].unique()
    )

    region = st.selectbox(
        "Region",
        dataset["Region"].unique()
    )

    monthly_quantity = st.number_input(
        "Monthly Quantity",
        min_value=0.0,
        value=500.0
    )

    purchase_frequency = st.number_input(
        "Purchase Frequency",
        min_value=0.0,
        value=15.0
    )

with col2:
    average_order = st.number_input(
        "Average Order",
        min_value=0.0,
        value=30.0
    )

    payment_delay_days = st.number_input(
        "Payment Delay Days",
        min_value=0.0,
        value=5.0
    )

    credit_limit = st.number_input(
        "Credit Limit",
        min_value=0.0,
        value=100000.0
    )

    relationship_years = st.number_input(
        "Relationship Years",
        min_value=0.0,
        value=10.0
    )

with col3:
    previous_month_sales = st.number_input(
        "Previous Month Sales",
        min_value=0.0,
        value=150000.0
    )

    current_month_sales = st.number_input(
        "Current Month Sales",
        min_value=0.0,
        value=170000.0
    )

    complaint_count = st.number_input(
        "Complaint Count",
        min_value=0,
        value=1
    )

# Automatic Growth Percentage
if previous_month_sales > 0:
    growth_percentage = (
        (current_month_sales - previous_month_sales)
        / previous_month_sales
    ) * 100
else:
    growth_percentage = 0.0

st.metric(
    "Automatically Calculated Growth %",
    f"{growth_percentage:.2f}%"
)

# Create input data
input_data = pd.DataFrame({
    "Monthly_Quantity": [monthly_quantity],
    "Purchase_Frequency": [purchase_frequency],
    "Average_Order": [average_order],
    "Payment_Delay_Days": [payment_delay_days],
    "Credit_Limit": [credit_limit],
    "Relationship_Years": [relationship_years],
    "Previous_Month_Sales": [previous_month_sales],
    "Current_Month_Sales": [current_month_sales],
    "Growth_Percentage": [growth_percentage],
    "Complaint_Count": [complaint_count],

    "Customer_Type_Dealer": [
        int(customer_type == "Dealer")
    ],

    "Customer_Type_Retailer": [
        int(customer_type == "Retailer")
    ],

    "Region_Coimbatore": [
        int(region == "Coimbatore")
    ],

    "Region_Dindigul": [
        int(region == "Dindigul")
    ],

    "Region_Erode": [
        int(region == "Erode")
    ],

    "Region_Madurai": [
        int(region == "Madurai")
    ],

    "Region_Salem": [
        int(region == "Salem")
    ],

    "Region_Tirunelveli": [
        int(region == "Tirunelveli")
    ],

    "Region_Trichy": [
        int(region == "Trichy")
    ]
})

# Match exact training feature order
input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)

st.divider()

# Prediction
if st.button(
    "🔮 Predict Customer Category",
    use_container_width=True
):

    prediction = model.predict(input_data)[0]

    reverse_mapping = {
        0: "Low Value",
        1: "Regular",
        2: "High Value"
    }

    predicted_category = reverse_mapping[prediction]

    st.success(
        f"Predicted Customer Category: {predicted_category}"
    )

    # Prediction Probability
    probabilities = model.predict_proba(input_data)[0]

    probability_table = pd.DataFrame({
        "Customer Category": [
            "Low Value",
            "Regular",
            "High Value"
        ],
        "Probability (%)": (
            probabilities * 100
        ).round(2)
    })

    st.subheader("Prediction Probability")

    st.dataframe(
        probability_table,
        use_container_width=True
    )

st.divider()

# Model Information
st.subheader("🤖 Model Information")

st.write("Final Model: Tuned Decision Tree")
st.write("Test Accuracy: 92.50%")
st.write("Number of Features: 19")