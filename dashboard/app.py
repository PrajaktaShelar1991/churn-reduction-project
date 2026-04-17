import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Churn Dashboard", layout="wide")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    accounts = pd.read_csv("../data/accounts.csv")
    events = pd.read_csv("../data/user_events.csv")
    feature = pd.read_csv("../data/feature_usage.csv")
    tickets = pd.read_csv("../data/support_tickets.csv")
    return accounts, events, feature, tickets

accounts, events, feature, tickets = load_data()

st.title("🤖 AI-Powered Churn Intelligence Dashboard")

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("🔍 Filters")

plan_filter = st.sidebar.multiselect("Select Plan", accounts["plan"].unique(), default=accounts["plan"].unique())
region_filter = st.sidebar.multiselect("Select Region", accounts["region"].unique(), default=accounts["region"].unique())
industry_filter = st.sidebar.multiselect("Select Industry", accounts["industry"].unique(), default=accounts["industry"].unique())

filtered_accounts = accounts[
    (accounts["plan"].isin(plan_filter)) &
    (accounts["region"].isin(region_filter)) &
    (accounts["industry"].isin(industry_filter))
]

# -----------------------------
# Merge Data
# -----------------------------
merged = filtered_accounts.merge(feature, on="account_id", how="left")

rage = events.groupby("account_id")["rage_click"].sum().reset_index()
merged = merged.merge(rage, on="account_id", how="left")

ticket_counts = tickets.groupby("account_id").size().reset_index(name="tickets")
merged = merged.merge(ticket_counts, on="account_id", how="left")

merged = merged.fillna(0)

# -----------------------------
# KPIs
# -----------------------------
total_accounts = len(merged)
churned_accounts = merged[merged["churned"] == 1].shape[0]
churn_rate = (churned_accounts / total_accounts) * 100 if total_accounts > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Accounts", total_accounts)
col2.metric("Churn Rate", f"{churn_rate:.1f}%")
col3.metric("Churned", churned_accounts)

st.divider()

# -----------------------------
# ML Model
# -----------------------------
st.subheader("🤖 Churn Prediction")

X = merged[["usage_count", "rage_click", "tickets"]]
y = merged["churned"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

merged["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

# Risk label
def risk_label(p):
    if p > 0.7:
        return "High Risk"
    elif p > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

merged["risk_segment"] = merged["churn_probability"].apply(risk_label)

# -----------------------------
# Doughnut Chart
# -----------------------------
def donut_chart(value, label):
    fig, ax = plt.subplots()
    ax.pie([value, 100 - value], labels=[label, ""], autopct='%1.1f%%')
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    return fig

# -----------------------------
# Behavioral Charts
# -----------------------------
st.subheader("📊 Behavioral Insights")

col1, col2, col3 = st.columns(3)

low_usage = merged[merged["usage_count"] < 5].shape[0] / total_accounts * 100 if total_accounts > 0 else 0
col1.pyplot(donut_chart(low_usage, "Low Adoption"))

high_rage = merged[merged["rage_click"] > 0].shape[0] / total_accounts * 100 if total_accounts > 0 else 0
col2.pyplot(donut_chart(high_rage, "High Rage"))

drop_off = 40
col3.pyplot(donut_chart(drop_off, "Drop-off"))

st.divider()

# -----------------------------
# AI Insights
# -----------------------------
st.subheader("🤖 AI Insights")

if low_usage > 40:
    st.warning("Low feature adoption is driving churn")

if high_rage > 30:
    st.warning("Rage clicks indicate UX issues in invoice upload")

if churn_rate > 10:
    st.error("Churn rate is above acceptable threshold")

# -----------------------------
# Risk Distribution
# -----------------------------
st.subheader("⚠️ Risk Segmentation")

risk_counts = merged["risk_segment"].value_counts()
st.bar_chart(risk_counts)

# -----------------------------
# Top Risk Accounts
# -----------------------------
st.subheader("🔴 Top At-Risk Accounts")

top_risk = merged.sort_values(by="churn_probability", ascending=False).head(5)

st.dataframe(top_risk[[
    "account_id",
    "plan",
    "usage_count",
    "rage_click",
    "tickets",
    "churn_probability",
    "risk_segment"
]])

# -----------------------------
# Recommendations
# -----------------------------
st.subheader("🚀 Recommended Actions")

st.markdown("""
- Target high-risk accounts with proactive outreach  
- Improve onboarding for low-usage users  
- Fix UX issues in invoice upload flow  
- Trigger alerts based on churn probability  
""")
