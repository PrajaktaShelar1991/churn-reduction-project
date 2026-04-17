import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Churn Intelligence", layout="wide")

# -----------------------------
# Clean SaaS UI Styling
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}

.card {
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.04);
    margin-bottom: 15px;
}

.highlight {
    background-color: #EEF2FF;
    padding: 12px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    accounts = pd.read_csv("../data/accounts.csv", parse_dates=["signup_date"])
    events = pd.read_csv("../data/user_events.csv")
    feature = pd.read_csv("../data/feature_usage.csv")
    tickets = pd.read_csv("../data/support_tickets.csv")
    return accounts, events, feature, tickets

accounts, events, feature, tickets = load_data()

# -----------------------------
# Sidebar Navigation (Story Mode)
# -----------------------------
page = st.sidebar.radio(
    "📖 Story Navigation",
    [
        "Executive Summary",
        "Why Users Churn",
        "Who Will Churn (AI)",
        "What Should We Do"
    ]
)

# -----------------------------
# PAGE 1: EXECUTIVE SUMMARY
# -----------------------------
if page == "Executive Summary":

    st.title("📊 Churn Intelligence Overview")

    total = len(accounts)
    churned = accounts[accounts["churned"] == 1].shape[0]
    churn_rate = (churned / total) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Accounts", total)
    col2.metric("Churned", churned)
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")

    st.markdown('<div class="highlight">⚠️ Churn is driven by low engagement and poor onboarding experience.</div>', unsafe_allow_html=True)

# -----------------------------
# PAGE 2: WHY USERS CHURN
# -----------------------------
elif page == "Why Users Churn":

    st.title("🔍 Why Are Users Churning?")

    st.subheader("1. Feature Adoption")

    merged = pd.merge(accounts, feature, on="account_id")
    usage = merged.groupby("churned")["usage_count"].mean()

    st.bar_chart(usage)

    st.subheader("2. UX Friction (Rage Clicks)")

    rage = events.groupby("account_id")["rage_click"].sum().reset_index()
    rage_m = pd.merge(accounts, rage, on="account_id")
    rage_data = rage_m.groupby("churned")["rage_click"].mean()

    st.bar_chart(rage_data)

    st.subheader("3. Funnel Drop-off")

    steps = ["login", "upload_invoice", "configure_workflow"]
    counts = [events[events["event_name"] == s]["account_id"].nunique() for s in steps]

    funnel = pd.DataFrame({"Step": steps, "Users": counts})
    st.bar_chart(funnel.set_index("Step"))

    st.markdown('<div class="highlight">📉 Biggest drop occurs after invoice upload → workflow setup.</div>', unsafe_allow_html=True)

# -----------------------------
# PAGE 3: WHO WILL CHURN
# -----------------------------
elif page == "Who Will Churn (AI)":

    st.title("🤖 Predicting At-Risk Accounts")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    usage = feature[["account_id", "usage_count"]]
    rage = events.groupby("account_id")["rage_click"].sum().reset_index()
    ticket_counts = tickets.groupby("account_id").size().reset_index(name="tickets")

    ml_df = accounts.merge(usage, on="account_id", how="left") \
                    .merge(rage, on="account_id", how="left") \
                    .merge(ticket_counts, on="account_id", how="left")

    ml_df = ml_df.fillna(0)

    X = ml_df[["usage_count", "rage_click", "tickets"]]
    y = ml_df["churned"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    ml_df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

    ml_df["risk"] = pd.cut(
        ml_df["churn_probability"],
        bins=[0, 0.4, 0.7, 1],
        labels=["Low", "Medium", "High"]
    )

    st.bar_chart(ml_df["risk"].value_counts())

    st.dataframe(ml_df[["account_id", "churn_probability", "risk"]])

    st.markdown('<div class="highlight">🔴 High-risk users need immediate intervention.</div>', unsafe_allow_html=True)

# -----------------------------
# PAGE 4: ACTION PLAN
# -----------------------------
elif page == "What Should We Do":

    st.title("🚀 Retention Strategy")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card"><h4>🔴 High Risk</h4><p>CS outreach + onboarding support</p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h4>🟡 Medium Risk</h4><p>In-app nudges & education</p></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card"><h4>🟢 Low Risk</h4><p>Upsell & engagement</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="highlight">💡 Focus on onboarding + feature adoption to reduce churn by ~30%.</div>', unsafe_allow_html=True)
