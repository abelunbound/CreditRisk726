import streamlit as st


# Function to display the sign-up form
def show_signup_form():
    st.markdown("## Sign Up")
    with st.form("signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        org_name = st.text_input("Organization Name")
        user_type = st.selectbox("User Type", ["Financial Institution", "Government", "Landlord", "Individual"])
        submitted = st.form_submit_button("Sign Up")

        if submitted:
            st.success(f"Sign Up successful for {email}!")
            # Here you would typically add logic to save the user information


# Function to display the login form
def show_login_form():
    st.markdown("## Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            st.success(f"Logged in as {email}!")
            # Here you would typically add logic to authenticate the user


# Function to display social login options
def show_social_login_options():
    st.markdown("## Or Log In With")
    st.button("Log In with Google")
    st.button("Log In with LinkedIn")


# Main function to run the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Sign Up", "Login"])

    if page == "Sign Up":
        show_signup_form()
    else:
        show_login_form()
        show_social_login_options()


if __name__ == "__main__":
    main()
