import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fairness Auditor", layout="wide")

st.title("⚖️ AI Fairness Auditor Dashboard")

# -------------------------
# DATA (simulated)
# -------------------------
np.random.seed(42)
n = 300

df = pd.DataFrame({
    "gender": np.random.choice(["Male", "Female"], n),
    "region": np.random.choice(["Urban", "Rural"], n),
    "income": np.random.randint(20000, 100000, n),
    "actual": np.random.choice([0, 1], n)
})

# biased prediction logic
df["predicted"] = np.where(
    (df["gender"] == "Male") & (np.random.rand(n) > 0.3), 1,
    np.where((df["gender"] == "Female") & (np.random.rand(n) > 0.6), 1, 0)
)

# -------------------------
# SIDEBAR CONFIG
# -------------------------
st.sidebar.header("⚙️ Configuration")

protected = st.sidebar.selectbox("Protected Attribute", ["gender", "region"])
target = "actual"
prediction = "predicted"

# -------------------------
# DATA AUDIT
# -------------------------
st.subheader("🔍 Data Audit")

col1, col2 = st.columns(2)

with col1:
    st.write("### Group Distribution")
    st.bar_chart(df[protected].value_counts())

with col2:
    st.write("### Missing Values")
    st.write(df.isnull().sum())

# -------------------------
# FAIRNESS METRICS
# -------------------------
st.subheader("⚖️ Fairness Metrics")

grouped = df.groupby(protected)

selection_rate = grouped[prediction].mean()

false_positive = grouped.apply(
    lambda x: ((x[prediction] == 1) & (x[target] == 0)).mean()
)

false_negative = grouped.apply(
    lambda x: ((x[prediction] == 0) & (x[target] == 1)).mean()
)

metrics_df = pd.DataFrame({
    "Selection Rate": selection_rate,
    "False Positive Rate": false_positive,
    "False Negative Rate": false_negative
})

st.dataframe(metrics_df)

# -------------------------
# VISUALIZATION
# -------------------------
st.subheader("📊 Visual Comparison")

fig, ax = plt.subplots()
metrics_df.plot(kind="bar", ax=ax)
ax.set_title(f"Fairness Metrics by {protected}")
st.pyplot(fig)

# -------------------------
# BIAS DETECTION
# -------------------------
st.subheader("🚨 Bias Detection")

gaps = metrics_df.max() - metrics_df.min()

st.write("### Metric Gaps")
st.write(gaps)

threshold = 0.1

for metric, value in gaps.items():
    if value > threshold:
        st.error(f"⚠️ Bias detected in {metric} (gap = {value:.2f})")
    else:
        st.success(f"✅ {metric} within acceptable range")

# -------------------------
# INSIGHTS
# -------------------------
st.subheader("🧠 Insights")

if gaps["Selection Rate"] > 0.1:
    st.write("The model favors one group significantly in approvals.")

if gaps["False Positive Rate"] > 0.1:
    st.write("Some groups are more likely to be incorrectly approved.")

if gaps["False Negative Rate"] > 0.1:
    st.write("Some groups are more likely to be unfairly rejected.")

st.write("Recommendation: Review features and consider fairness-aware modeling.")
