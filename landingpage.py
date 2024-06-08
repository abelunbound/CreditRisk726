import streamlit as st
from PIL import Image

# Load logo image
logo = Image.open('path_to_logo.png')  # Replace with your logo file path

# Set page configuration
st.set_page_config(page_title="RiskAwareAI", page_icon=logo)

# Header with logo and navigation links
st.image(logo, width=100)
st.markdown("""
<nav>
    <a href="#home">Home</a> |
    <a href="#features">Features</a> |
    <a href="#pricing">Pricing</a> |
    <a href="#about-us">About Us</a> |
    <a href="/auth">Sign Up / Login</a>
</nav>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
# Welcome to RiskAwareAI
Our mission is to provide innovative AI-driven solutions for financial risk measurement benefiting organizations and individuals, particularly the 5 million credit invisible population in the UK and millions more globally.
""")
if st.button("Get Started"):
    st.experimental_rerun()  # This would redirect to a specific section or page if needed

# Features Overview
st.markdown("""
## Features
### ForseeRisk
Predicts the risk of default for applicants with no credit history using deep learning models. Supports credit and loan decisioning, and recommends adaptive payment plans to minimize defaults.

### GoodFaith
Provides a global view of any applicant's adverse credit, liability, court judgment debt, and other risks through permissioned access to financial data in over 30 countries. Used by governments for visa applications and landlords for tenant screening.
""")

# Testimonials
st.markdown("""
## Testimonials
> "RiskAwareAI helped us streamline our credit decision process and significantly reduced our default rates." - John Doe, Financial Analyst

> "GoodFaith is a game-changer for verifying the financial backgrounds of international students and professionals." - Jane Smith, Property Manager
""")

# Footer
st.markdown("""
<footer>
    <p>Contact us: info@riskawareai.com</p>
    <p>Follow us on <a href="https://twitter.com/yourprofile">Twitter</a> | <a href="https://www.linkedin.com/yourprofile">LinkedIn</a></p>
    <p><a href="privacy_policy_url">Privacy Policy</a> | <a href="terms_of_service_url">Terms of Service</a></p>
</footer>
""", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
<style>
nav {
    font-size: 18px;
    margin-bottom: 20px;
}
nav a {
    text-decoration: none;
    color: #4CAF50;
    padding: 0 15px;
}
nav a:hover {
    text-decoration: underline;
}
footer {
    text-align: center;
    margin-top: 50px;
}
footer p {
    margin: 5px 0;
}
footer a {
    text-decoration: none;
    color: #4CAF50;
}
footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)
