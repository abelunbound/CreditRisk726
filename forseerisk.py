import streamlit as st
import pandas as pd
from datetime import datetime


# Function to display the overview page
def overview_page():
    st.title("ForseeRisk Overview")
    st.markdown("""
    ForseeRisk predicts the risk of default for applicants with no credit history using advanced deep learning models. 
    This tool helps financial institutions, governments, and landlords make informed decisions.
    """)
    # Example statistics
    st.markdown("""
    ### Statistics:
    - **Total Assessments Conducted:** 1,234
    - **Average Risk Score:** 65%
    - **Assessments in the Last Month:** 123
    """)


# Function to display the new risk assessment form
def new_assessment_form():
    st.title("New Risk Assessment")
    st.markdown("Fill in the details below to start a new risk assessment.")

    with st.form("assessment_form"):
        name = st.text_input("Applicant Name")
        dob = st.date_input("Date of Birth", datetime(1990, 1, 1))
        nationality = st.selectbox("Nationality", ["United Kingdom", "United States", "Canada", "Australia", "Other"])
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Student"])
        financial_docs = st.file_uploader("Upload Financial Documents", accept_multiple_files=True)

        submitted = st.form_submit_button("Submit")

        if submitted:
            # Here you would typically handle form submission, e.g., save data and start assessment
            st.success("Risk assessment started for {}.".format(name))
            # For simplicity, we'll redirect to a mock results page with static data
            st.session_state['results'] = {
                'name': name,
                'dob': dob,
                'nationality': nationality,
                'employment_status': employment_status,
                'risk_score': 75,  # Mock risk score
                'analysis': {
                    'credit_history': 'No credit history found.',
                    'income_stability': 'Stable income with recent employment.',
                    'debt_to_income_ratio': 'Low debt-to-income ratio.'
                },
                'recommendations': 'Consider offering a small loan with flexible repayment options.'
            }
            st.experimental_rerun()


# Function to display the results page
def results_page():
    st.title("Risk Assessment Results")
    if 'results' in st.session_state:
        results = st.session_state['results']
        st.markdown(f"### Applicant: {results['name']}")
        st.markdown(f"**Date of Birth:** {results['dob']}")
        st.markdown(f"**Nationality:** {results['nationality']}")
        st.markdown(f"**Employment Status:** {results['employment_status']}")

        st.markdown("### Risk Score")
        st.progress(results['risk_score'] / 100.0)

        st.markdown("### Detailed Analysis")
        analysis = results['analysis']
        st.markdown(f"- **Credit History:** {analysis['credit_history']}")
        st.markdown(f"- **Income Stability:** {analysis['income_stability']}")
        st.markdown(f"- **Debt to Income Ratio:** {analysis['debt_to_income_ratio']}")

        st.markdown("### Recommendations")
        st.markdown(results['recommendations'])
    else:
        st.markdown("No results available. Please complete a new risk assessment.")


# Main function to run the app
def main():
    st.sidebar.title("ForseeRisk Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "New Risk Assessment", "Results"])

    if page == "Overview":
        overview_page()
    elif page == "New Risk Assessment":
        new_assessment_form()
    else:
        results_page()


if __name__ == "__main__":
    main()
