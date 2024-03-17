
# Import Libraries and Dependencies
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

#Dependencies for Feature Extraction and Engineering
# SKLEARN is installed from scikit-learn as sklearn is deperecated
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

#Save models
import joblib


#Preprocessed data used for training imported to help structure input from loan applicant

##########################################################################
df_dummies1 = pd.read_csv("df_dummies1_09_march_2024.csv")

# Spliting dataset into train and test version

#Remove Target Variable column, assign resultant dataframe with only indpendent variables to "x"
x = df_dummies1.drop('Credit Risk', 1).values

#Select Target variable column, assign it to "y"
y = df_dummies1['Credit Risk'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30,random_state=0)


# Scaling the dataset
scaled_x = StandardScaler()
x_train = scaled_x.fit_transform(x_train)
x_test = scaled_x.transform(x_test)

##########################################################################


# Preprocess & Prediction Functions

# Function to preprocess input data
def preprocess_input(input_df):
    # Perform any necessary data preprocessing (label encoding, scaling, etc.)

    # 1. copy dataframe from input fields
    df_selected_test = input_df.copy()

    # 2.
    # Second list with names of columns for one-hot encoding of 9 categorical variables for model prediction
    # Final dummy will use 9 categorical variables, and 3 numerical variable and 1 target variable

    col_cat_names=["Status of existing checking account","Purpose",\
    "Savings account/bonds","Present employment since",\
    "Personal status and sex","Other debtors / guarantors","Property","Other installment plans","Housing"]


    # Use one-hot encoding to create dummy variables for 9 categorical variables created

    class Preprocessor:
        def __init__(self, df_selected_test, col_cat_names):
            self.df_selected_test = df_selected_test
            self.col_cat_names = col_cat_names

        def one_hot_encode(self):
            for attr in self.col_cat_names:
                self.df_selected_test = self.df_selected_test.merge(pd.get_dummies(self.df_selected_test[attr], prefix=attr), left_index=True, right_index=True)
                self.df_selected_test.drop(attr, axis=1, inplace=True)

    preprocessor = Preprocessor(df_selected_test, col_cat_names)
    preprocessor.one_hot_encode()
    df_dummies_test = preprocessor.df_selected_test


    # 3. Before the model can predict, it has to have the same number of features as the x_test used in training the model. Input data has
    # 12, the x_test has 44

    missing_columns = set(df_dummies1.columns) - set(df_dummies_test.columns)

    # 4. Get a list of column names in the order in which they appear from the dataframe used in creating x_test
    column_names_in_order = list(df_dummies1.columns)

    # 5. Replace all missing columns with 0
    for column in missing_columns:
        df_dummies_test[column] = 0


    # List of column names in the desired order

    # Reorder the DataFrame columns
    df_dummies_test = df_dummies_test[column_names_in_order]


    # 6. drop "Credit Risk" column so its only 44 columns like in x_test?
    df_dummies_test = df_dummies_test.drop(columns=['Credit Risk'])

    # 7. # x_test has column names replaced with index numbers, do the same for df_dummies_test
    df_dummies_test.columns = range(len(df_dummies_test.columns))


    # 8. Scaling the loan applicant data

    x_applicant_test = scaled_x.transform(df_dummies_test)


    return x_applicant_test

# Function to make loan prediction
def predict_loan_approval(input_df):
    # Preprocess the input data
    processed_data = preprocess_input(input_df)

    # Load the model from the file
    loaded_model = joblib.load('xgboost_credit_risk_no_credit_history.pkl')

    # Make predictions
    prediction = loaded_model.predict(processed_data)

    return prediction

def probability_of_default(input_df):
    # Preprocess the input data
    processed_data = preprocess_input(input_df)

    # Load the model from the file
    loaded_model = joblib.load('xgboost_credit_risk_no_credit_history.pkl')

    # Make predictions

    p_default = loaded_model.predict_proba(processed_data)
    p_default = pd.DataFrame(p_default, columns=['Probability this applicant will default:', 'GoodRisk'])
    p_default = p_default.multiply(100)
    p_default = p_default.applymap(lambda x: f'{x:.2f}%')
    return p_default



# Streamlit UI
def main():
    st.title("UNBOUND: The Credit Risk Measurement System for Immigrants")
    st.subheader("Just arrived in the UK? Need between £100 and £2,000 credit? ")
    st.markdown("Get your UNBOUND Credit Card today; check your eligiblity below! ")
    # Dropdown for 'Purpose' field
    purpose_options = ["car (new)", "car (used)", "furniture/equipment", "radio/television",
                       "domestic appliances", "repairs", "education", 'vacation', 'retraining',
                       'business', 'others']
    purpose = st.selectbox('Purpose', purpose_options)

    # Collect input from user using sliders and dropdown
    duration = st.slider('Duration in months', 1, 36, 12)
    installment_rate = st.slider('Installment rate in % of disposable income', 0, 10, 5)
    age = st.slider('Age in years', 18, 80, 30)

    # Dropdown for 'Checking account' field
    checking_account = ["no checking account", "<0 DM", "0 <= <200 DM", ">= 200 DM "]
    checking = st.selectbox('Checking Account Status', checking_account)



    # Dropdown for 'Savings account/bond' field
    savings_account = ["< £100", "100 <= <500 DM", "500 <= < 1000 DM", ">= 1000 DM"]
    savings = st.selectbox('Savings account/bonds', savings_account)

    # Dropdown for 'Present employment since' field
    present_employment = [">=7 years", "4<= <7 years", "1<= < 4 years", "<1 years", "unemployed"]
    employment_duration = st.selectbox('Present employment since', present_employment)

    # Dropdown for 'Personal status and sex' field
    personal_status = ["female:single", "male:married/widowed", "male:single", "female:divorced/separated/married",
                       "male:divorced/separated"]
    status = st.selectbox('Personal status and sex', personal_status)

    # Dropdown for 'Other debtors / guarantors' field
    other_debtors_guarantor = ["none", "co-applicant", "guarantor"]
    other_debtors = st.selectbox('Other debtors / guarantors', other_debtors_guarantor)

    # Dropdown for 'Property' field
    property_field = ["real estate", "savings agreement/life insurance", "car or other", "unknown / no property"]
    property_owned = st.selectbox('Property', property_field)

    # Dropdown for 'Other installment plans' field
    installment_plan = ["none", "store", "bank"]
    installment = st.selectbox('Other installment plans', installment_plan)

    # Dropdown for 'Housing' field
    housing_type = ["for free", "own", "rent"]
    housing = st.selectbox('Housing', housing_type)

    # Other fields collected as text inputs
    # fields = ['', '', '', '']

    user_input = {}
    # for field in fields:
    #     user_input[field] = st.text_input(field)

    # Add the slider and dropdown values to the user input dictionary
    user_input['Duration in month'] = duration
    user_input['Installment rate in percentage of disposable income'] = installment_rate
    user_input['Age in years'] = age
    user_input['Purpose'] = purpose
    user_input['Status of existing checking account'] = checking
    user_input['Savings account/bonds'] = savings
    user_input['Present employment since'] = employment_duration
    user_input['Personal status and sex'] = status
    user_input['Other debtors / guarantors'] = other_debtors
    user_input['Property'] = property_owned
    user_input['Other installment plans'] = installment
    user_input['Housing'] = housing

    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Display user input
    st.subheader("User Input:")
    st.write(input_df)

    #Preprocess input data

    # Make prediction using input data

    if st.button("Check eligibility"):
        prediction = predict_loan_approval(input_df)
        prob_default = probability_of_default(input_df)

        # Interpret the prediction
        predicted_label = 'Your £1000 credit has been approved' if prediction[
                                                                       0] == 1 else 'Your £1000 credit application has been rejected'

        st.subheader("Outcome:")
        st.write(predicted_label)
        st.write(prob_default)

if __name__ == "__main__":
    main()
