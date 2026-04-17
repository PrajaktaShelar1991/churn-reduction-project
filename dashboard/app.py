import streamlit as st
import pandas as pd

st.set_page_config(page_title="Churn Analytics Dashboard", layout="wide")

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

# -----------------------------
# Title
# -----------------------------
st.title("📉 Customer Churn Analytics Dashboard")
st.markdown("Simulating product analytics insights (Pendo-style)")

# -----------------------------
# KPI Section
# -----------------------------
total_accounts = len(accounts)
churned_accounts = accounts[accounts["churned"] == 1].shape[0]
churn_rate = (churned_accounts / total_accounts) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Accounts", total_accounts)
col2.metric("Churned Accounts", churned_accounts)
col3.metric("Churn Rate (%)", f"{churn_rate:.2f}")

st.divider()

# -----------------------------
# Feature Usage vs Churn
# -----------------------------
st.subheader("📊 Feature Usage vs Churn")

merged = pd.merge(accounts, feature, on="account_id")

usage_by_churn = merged.groupby("churned")["usage_count"].mean()

st.bar_chart(usage_by_churn)

st.caption("Insight: Lower feature usage correlates with churn")

# -----------------------------
# Rage Click Analysis
# -----------------------------
st.subheader("⚠️ Rage Click Analysis")

rage_clicks = events.groupby("account_id")["rage_click"].sum().reset_index()
rage_merged = pd.merge(accounts, rage_clicks, on="account_id")

rage_by_churn = rage_merged.groupby("churned")["rage_click"].mean()

st.bar_chart(rage_by_churn)

st.caption("Insight: Higher rage clicks indicate UX friction")

# -----------------------------
# Funnel Analysis
# -----------------------------
st.subheader("📉 Funnel Drop-off Analysis")

funnel_steps = ["login", "upload_invoice", "configure_workflow"]

funnel_counts = []
for step in funnel_steps:
    count = events[events["event_name"] == step]["account_id"].nunique()
    funnel_counts.append(count)

funnel_df = pd.DataFrame({
    "Step": funnel_steps,
    "Users": funnel_counts
})

st.bar_chart(funnel_df.set_index("Step"))

st.caption("Insight: Major drop between upload_invoice → configure_workflow")

# -----------------------------
# Support Tickets Analysis
# -----------------------------
st.subheader("🎫 Support Ticket Trends")

ticket_counts = tickets.groupby("account_id").size().reset_index(name="tickets")

ticket_merged = pd.merge(accounts, ticket_counts, on="account_id", how="left")
ticket_merged["tickets"] = ticket_merged["tickets"].fillna(0)

tickets_by_churn = ticket_merged.groupby("churned")["tickets"].mean()

st.bar_chart(tickets_by_churn)

st.caption("Insight: Churned users raise more support tickets")

# -----------------------------
# Churned Accounts Table
# -----------------------------
st.subheader("📋 At-Risk / Churned Accounts")

churned_df = accounts[accounts["churned"] == 1]
st.dataframe(churned_df)

# -----------------------------
# Key Insights Section
# -----------------------------
st.subheader("🧠 Key Insights")

st.markdown("""
- 🔴 Low feature adoption strongly correlates with churn  
- ⚠️ Rage clicks highlight UX issues in invoice upload  
- 📉 Major drop-off in onboarding funnel  
- 🎫 High support dependency among churned users  
""")

# -----------------------------
# Recommendations
# -----------------------------
st.subheader("🚀 Recommendations")

st.markdown("""
1. Improve onboarding with guided walkthroughs  
2. Fix invoice upload UX issues  
3. Introduce feature adoption nudges  
4. Implement churn prediction & alerts  
""")
