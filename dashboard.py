import streamlit as st
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="RiskAwareAI Dashboard", page_icon="📊", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("""
- [Dashboard](#)
- [ForseeRisk](#)
- [GoodFaith](#)
- [Reports](#)
- [Settings](#)
- [Support](#)
""")

# Main Dashboard Home
st.title("RiskAwareAI Dashboard")

# Welcome Message
user_name = "John Doe"  # This could be dynamically set based on the logged-in user
st.markdown(f"## Welcome back, {user_name}!")
st.markdown(f"### Today's Date: {datetime.now().strftime('%A, %B %d, %Y')}")

# Quick Overview
st.markdown("### Quick Overview")
st.markdown("""
<div style='background-color: #e8f5e9; padding: 10px; border-radius: 5px;'>
    <h4>Recent Activity</h4>
    <ul>
        <li>Risk assessment for Jane Smith completed.</li>
        <li>Data uploaded for new applicants.</li>
        <li>Report generated for April 2024.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Quick Actions
st.markdown("### Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("New Risk Assessment"):
        st.write("Redirect to New Risk Assessment page.")
        # Add logic to redirect to the New Risk Assessment page

with col2:
    if st.button("Upload Data"):
        st.write("Redirect to Upload Data page.")
        # Add logic to redirect to the Upload Data page

with col3:
    if st.button("View Reports"):
        st.write("Redirect to View Reports page.")
        # Add logic to redirect to the View Reports page

# Custom CSS for styling
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f0f4c3;
        color: black;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4caf50;
    }
    .stTitle {
        color: #2e7d32;
    }
    .stHeader {
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)
