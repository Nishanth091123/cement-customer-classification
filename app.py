import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Cement Customer Classification", page_icon="📊", layout="wide")

BASE = Path(__file__).resolve().parent

def first_existing(*names):
    for name in names:
        p = BASE / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing required file. Tried: {', '.join(names)}")

DATA_FILE = first_existing("cement_customer_classification_1000.csv", "cement_customer_classification_1000(1).csv")
MODEL_FILE = first_existing("cement_customer_final_model.pkl")
FEATURE_FILE = first_existing("cement_customer_feature_columns.pkl")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_features():
    with open(FEATURE_FILE, "rb") as f:
        return pickle.load(f)

dataset = load_data()
model = load_model()
feature_columns = load_features()

st.markdown("""
<style>
    .stApp { background: #f4f7f9; }
    .block-container { max-width: 1320px; padding-top: 1.2rem; padding-bottom: 2.2rem; }
    header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
    #MainMenu, footer { visibility: hidden; }

    .hero {
        background: linear-gradient(120deg, #102a43 0%, #164e63 58%, #0f766e 100%);
        border-radius: 18px; padding: 27px 32px; margin-bottom: 22px;
        box-shadow: 0 10px 28px rgba(16,42,67,.12);
    }
    .eyebrow { color:#a7f3d0; font-size:.76rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
    .hero h1 { color:#ffffff; font-size:2.05rem; margin:.35rem 0 .35rem; letter-spacing:-.02em; }
    .hero p { color:#e2edf3; margin:0; font-size:.98rem; max-width:900px; }
    .section-title { font-size:1.08rem; font-weight:750; color:#16324a; margin:.25rem 0 .8rem; }

    div[data-testid="stMetric"] {
        background:#ffffff; border:1px solid #dce5ea; border-radius:14px; padding:13px 16px;
        box-shadow:0 4px 14px rgba(16,42,67,.04);
    }
    div[data-testid="stMetricLabel"] { color:#647783; }
    div[data-testid="stMetricValue"] { color:#102a43; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:#ffffff; border-radius:16px; border-color:#dce5ea !important;
        box-shadow:0 5px 18px rgba(16,42,67,.04);
    }
    div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input {
        background:#f8fafb; border-color:#d8e2e8;
    }
    .stButton > button {
        background:linear-gradient(90deg,#0f766e,#167d89); color:white; border:0; border-radius:10px;
        min-height:3rem; font-weight:750; letter-spacing:.01em;
    }
    .stButton > button:hover { background:#0b625d; color:white; border:0; }

    .result-card { border-radius:16px; padding:22px 24px; color:white; min-height:150px; box-shadow:0 8px 22px rgba(16,42,67,.10); }
    .result-label { color:rgba(255,255,255,.82); font-size:.8rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }
    .result-value { font-size:2rem; font-weight:800; margin:.35rem 0; }
    .result-copy { color:rgba(255,255,255,.92); font-size:.92rem; }
    .insight { padding:14px 16px; border-radius:8px; color:#263640; }
    .model-card {
        margin-top:16px; background:#ffffff; border:1px solid #dce5ea; border-radius:12px;
        padding:14px 16px; color:#263640; box-shadow:0 4px 14px rgba(16,42,67,.04);
    }
    .model-title { font-size:.78rem; color:#667985; text-transform:uppercase; letter-spacing:.08em; font-weight:700; margin-bottom:5px; }
    .model-main { font-weight:750; color:#16324a; }
    hr { border-color:#dce5ea; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Customer Segmentation • Sales Intelligence</div>
  <h1>📊 Cement Customer Classification</h1>
  <p>Evaluate cement customer behaviour, sales momentum and payment profile to support smarter customer prioritisation.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Customer Assessment</div>', unsafe_allow_html=True)
with st.container(border=True):
    a, b, c = st.columns(3, gap="large")
    with a:
        st.caption("CUSTOMER PROFILE")
        customer_type = st.selectbox("Customer Type", dataset["Customer_Type"].dropna().unique())
        region = st.selectbox("Region", dataset["Region"].dropna().unique())
        relationship_years = st.number_input("Relationship Years", min_value=0.0, value=10.0, step=1.0)
        complaint_count = st.number_input("Complaint Count", min_value=0, value=1, step=1)
    with b:
        st.caption("PURCHASE BEHAVIOUR")
        monthly_quantity = st.number_input("Monthly Quantity", min_value=0.0, value=500.0, step=10.0)
        purchase_frequency = st.number_input("Purchase Frequency", min_value=0.0, value=15.0, step=1.0)
        average_order = st.number_input("Average Order", min_value=0.0, value=30.0, step=1.0)
        credit_limit = st.number_input("Credit Limit", min_value=0.0, value=100000.0, step=5000.0)
    with c:
        st.caption("SALES & PAYMENT")
        payment_delay_days = st.number_input("Payment Delay Days", min_value=0.0, value=5.0, step=1.0)
        previous_month_sales = st.number_input("Previous Month Sales", min_value=0.0, value=150000.0, step=5000.0)
        current_month_sales = st.number_input("Current Month Sales", min_value=0.0, value=170000.0, step=5000.0)

growth_percentage = ((current_month_sales - previous_month_sales) / previous_month_sales * 100) if previous_month_sales > 0 else 0.0

g1, g2, g3 = st.columns(3)
g1.metric("Sales Growth", f"{growth_percentage:.2f}%", delta=f"{growth_percentage:.2f}% vs previous month")
g2.metric("Current Sales", f"₹{current_month_sales:,.0f}")
g3.metric("Payment Delay", f"{payment_delay_days:.0f} days")

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
    "Customer_Type_Dealer": [int(customer_type == "Dealer")],
    "Customer_Type_Retailer": [int(customer_type == "Retailer")],
    "Region_Coimbatore": [int(region == "Coimbatore")],
    "Region_Dindigul": [int(region == "Dindigul")],
    "Region_Erode": [int(region == "Erode")],
    "Region_Madurai": [int(region == "Madurai")],
    "Region_Salem": [int(region == "Salem")],
    "Region_Tirunelveli": [int(region == "Tirunelveli")],
    "Region_Trichy": [int(region == "Trichy")],
}).reindex(columns=feature_columns, fill_value=0)

predict = st.button("Analyse Customer Value", use_container_width=True, type="primary")

if predict:
    prediction = model.predict(input_data)[0]
    reverse_mapping = {0: "Low Value", 1: "Regular", 2: "High Value"}
    predicted_category = reverse_mapping[prediction]
    probabilities = model.predict_proba(input_data)[0]
    probability_table = pd.DataFrame({
        "Customer Category": ["Low Value", "Regular", "High Value"],
        "Probability (%)": (probabilities * 100).round(2),
    })
    top_probability = float(probabilities.max() * 100)

    segment_style = {
        "Low Value": {
            "card": "linear-gradient(135deg,#991b1b,#dc2626)",
            "soft": "#fef2f2",
            "accent": "#dc2626",
            "msg": "Review account: examine purchase frequency, payment behaviour and opportunities for reactivation."
        },
        "Regular": {
            "card": "linear-gradient(135deg,#c2410c,#f59e0b)",
            "soft": "#fff7ed",
            "accent": "#f59e0b",
            "msg": "Development account: monitor purchase momentum and identify opportunities to increase engagement."
        },
        "High Value": {
            "card": "linear-gradient(135deg,#166534,#16a34a)",
            "soft": "#f0fdf4",
            "accent": "#16a34a",
            "msg": "Priority account: focus on retention, relationship strength and growth opportunities."
        }
    }
    style = segment_style[predicted_category]

    st.write("")
    st.markdown('<div class="section-title">Assessment Result</div>', unsafe_allow_html=True)
    r1, r2 = st.columns([1, 1.55], gap="large")

    with r1:
        st.markdown(f"""
        <div class="result-card" style="background:{style['card']};">
          <div class="result-label">Predicted Customer Segment</div>
          <div class="result-value">{predicted_category}</div>
          <div class="result-copy">Confidence: <b>{top_probability:.0f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown(
            f"""<div class="insight" style="background:{style['soft']}; border-left:4px solid {style['accent']};">
            <b>Recommended Action:</b> {style['msg']}
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="model-card">
          <div class="model-title">Model Performance</div>
          <div class="model-main">Tuned Decision Tree &nbsp;•&nbsp; Test Accuracy: 92.50%</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown("**Prediction Probability**")

        # Streamlit's native bar chart does not support a different fixed color
        # for each category reliably, so render a compact HTML probability view.
        colors = {
            "Low Value": "#dc2626",
            "Regular": "#f59e0b",
            "High Value": "#16a34a"
        }
        bars = []
        for _, row in probability_table.iterrows():
            category = row["Customer Category"]
            value = float(row["Probability (%)"])
            bars.append(
                f'<div style="margin:0 0 18px 0;">'
                f'<div style="display:flex;justify-content:space-between;'
                f'margin-bottom:6px;font-size:.88rem;color:#526672;">'
                f'<span>{category}</span>'
                f'<b style="color:#16324a;">{value:.2f}%</b>'
                f'</div>'
                f'<div style="height:18px;background:#e8eef2;border-radius:9px;overflow:hidden;">'
                f'<div style="width:{value}%;height:100%;background:{colors[category]};'
                f'border-radius:9px;"></div>'
                f'</div></div>'
            )

        probability_html = (
            '<div style="background:#ffffff;border:1px solid #dce5ea;border-radius:14px;'
            'padding:22px 22px 8px;box-shadow:0 4px 14px rgba(16,42,67,.04);">'
            + "".join(bars)
            + '</div>'
        )
        st.markdown(probability_html, unsafe_allow_html=True)

st.divider()
st.caption("Cement Customer Classification • Cement Customer Classification")
