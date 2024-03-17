import streamlit as st


# Dependencies for main script
# Import Libraries and Dependencies

import warnings
warnings.filterwarnings('ignore')

# Dependencies for data cleaning, analysis and manipulation
import pandas as pd
import numpy as np
import statistics
from scipy import stats
from math import floor,ceil
import statsmodels.api as sm

#Dependencies for Data Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# import scikitplot as skplt

#Dependencies for Feature Extraction and Engineering

# SKLEARN is installed from scikit-learn as sklearn is deperecated
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


### Dependencies for five (5) Model Creation

#Dependencies for Logistic Regression
from sklearn.linear_model import LogisticRegression

#Dependencies for Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

#Dependencies for Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

#Dependencies for Support Vector Machines
from sklearn.svm import SVC

#Dependencies for K-Nearest Neighbours (KNN)
# from sklearn.neighbors import KNeighborsClassifier

#Dependencies for Model Evaluation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, accuracy_score, classification_report, roc_curve,auc, f1_score, precision_score, recall_score, roc_auc_score
import sklearn.metrics as metrics

#Save models
import joblib


# # Data Collection
# # Data collection from data source
#
# german_credit_data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",sep=" ",header=None)
#
# column_headers = ["Status of existing checking account","Duration in month","Credit history",\
#          "Purpose","Credit amount","Savings account/bonds","Present employment since",\
#          "Installment rate in percentage of disposable income","Personal status and sex",\
#          "Other debtors / guarantors","Present residence since","Property","Age in years",\
#         "Other installment plans","Housing","Number of existing credits at this bank",\
#         "Job","Number of people being liable to provide maintenance for","Telephone","foreign worker","Cost Matrix(Risk)"]
#
# german_credit_data.columns = column_headers
#
# #Save as CSV file
# german_credit_data.to_csv("germancreditdata_6_mar_2024.csv",index=False)
#
# #Read data from CSV file
# ger_credit_df = pd.read_csv("germancreditdata_6_mar_2024.csv")
#
# #### Data Preprocessing
#
# # The current CSV file collected contains keys like "A14, A61" for each of the columns
# # UCI provides a description of each of this keys.
# # To proceed we need to map this these keys/entries to their respective descriptions by creating
# # a dictionary with a key-value pair for each feature. The key represents current entry, while values represent description
# # and using the .map() iterative method. The map() method creates a new array populated with
# # the results of calling a provided function on every element in the calling array.
#
#
#
# #Key-Value Pair for 13 categorical features
# Status_of_existing_checking_account={'A14':"no checking account",'A11':"<0 DM", 'A12': "0 <= <200 DM",'A13':">= 200 DM "}
# Credit_history={"A34":"critical account","A33":"delay in paying off","A32":"existing credits paid back duly till now","A31":"all credits at this bank paid back duly","A30":"no credits taken"}
# Purpose={"A40" : "car (new)", "A41" : "car (used)", "A42" : "furniture/equipment", "A43" :"radio/television" , "A44" : "domestic appliances", "A45" : "repairs", "A46" : "education", 'A47' : 'vacation','A48' : 'retraining','A49' : 'business','A410' : 'others'}
# Saving_account={"A65" : "no savings account","A61" :"<100 DM","A62" : "100 <= <500 DM","A63" :"500 <= < 1000 DM", "A64" :">= 1000 DM"}
# Present_employment={'A75':">=7 years", 'A74':"4<= <7 years",  'A73':"1<= < 4 years", 'A72':"<1 years",'A71':"unemployed"}
# Personal_status_and_sex={ 'A95':"female:single",'A94':"male:married/widowed",'A93':"male:single", 'A92':"female:divorced/separated/married", 'A91':"male:divorced/separated"}
# Other_debtors_guarantors={'A101':"none", 'A102':"co-applicant", 'A103':"guarantor"}
# Property={'A121':"real estate", 'A122':"savings agreement/life insurance", 'A123':"car or other", 'A124':"unknown / no property"}
# Other_installment_plans={'A143':"none", 'A142':"store", 'A141':"bank"}
# Housing={'A153':"for free", 'A152':"own", 'A151':"rent"}
# Job={'A174':"management/ highly qualified employee", 'A173':"skilled employee / official", 'A172':"unskilled - resident", 'A171':"unemployed/ unskilled  - non-resident"}
# Telephone={'A192':"yes", 'A191':"none"}
# foreign_worker={'A201':"yes", 'A202':"no"}
# risk={1:"Good Risk", 2:"Bad Risk"}
#
# #Using Map function to replace values in the columns
# ger_credit_df["Status of existing checking account"]=ger_credit_df["Status of existing checking account"].map(Status_of_existing_checking_account)
# ger_credit_df["Credit history"]=ger_credit_df["Credit history"].map(Credit_history)
# ger_credit_df["Purpose"]=ger_credit_df["Purpose"].map(Purpose)
# ger_credit_df["Savings account/bonds"]=ger_credit_df["Savings account/bonds"].map(Saving_account)
# ger_credit_df["Present employment since"]=ger_credit_df["Present employment since"].map(Present_employment)
# ger_credit_df["Personal status and sex"]=ger_credit_df["Personal status and sex"].map(Personal_status_and_sex)
# ger_credit_df["Other debtors / guarantors"]=ger_credit_df["Other debtors / guarantors"].map(Other_debtors_guarantors)
# ger_credit_df["Property"]=ger_credit_df["Property"].map(Property)
# ger_credit_df["Other installment plans"]=ger_credit_df["Other installment plans"].map(Other_installment_plans)
# ger_credit_df["Housing"]=ger_credit_df["Housing"].map(Housing)
# ger_credit_df["Job"]=ger_credit_df["Job"].map(Job)
# ger_credit_df["Telephone"]=ger_credit_df["Telephone"].map(Telephone)
# ger_credit_df["foreign worker"]=ger_credit_df["foreign worker"].map(foreign_worker)
# ger_credit_df["Cost Matrix(Risk)"]=ger_credit_df["Cost Matrix(Risk)"].map(risk)
#
#
# #Rename Column
#
# ger_credit_df.rename(columns = {'Cost Matrix(Risk)':'Credit Risk'}, inplace = True)
#
# # Removing outliers with the IQR method
#
# class OutlierDetector:
#     def __init__(self, variables, ger_credit_df):
#         self.variables = variables
#         self.ger_credit_df = ger_credit_df
#
#     def detect_outliers(self):
#         for v in self.variables:
#             subset_0 = self.ger_credit_df[self.ger_credit_df["Credit Risk"] == "Good Risk"]
#             subset_1 = self.ger_credit_df[self.ger_credit_df["Credit Risk"] == "Bad Risk"]
#
#             q75,q25 = np.percentile(subset_0.loc[:, v],[75, 25])
#             interval_q = q75 - q25
#             max_value = q75 + (1.5 * interval_q)
#             min_value = q25 - (1.5 * interval_q)
#
#             for i in range(1, len(self.ger_credit_df)):
#                 if (self.ger_credit_df.loc[i, v] < min_value) | (self.ger_credit_df.loc[i, v] > max_value):
#                     self.ger_credit_df.loc[i, v] = statistics.mean(self.ger_credit_df[v])
#
#             q75,q25 = np.percentile(subset_1.loc[:, v],[75, 25])
#             interval_q = q75 - q25
#             max_value = q75 + (1.5 * interval_q)
#             min_value = q25 - (1.5 * interval_q)
#
#             for i in range(1, len(self.ger_credit_df)):
#                 if (self.ger_credit_df.loc[i, v] < min_value) | (self.ger_credit_df.loc[i, v] > max_value):
#                     self.ger_credit_df.loc[i, v] = statistics.mean(self.ger_credit_df[v])
#
# # Instantiate the class and run the function
# variables = ["Credit amount","Duration in month",
#               "Installment rate in percentage of disposable income",
#               "Age in years","Present residence since",
#               "Number of existing credits at this bank",
#               "Number of people being liable to provide maintenance for"]
# detector = OutlierDetector(variables, ger_credit_df)
# detector.detect_outliers()
#
#
# # xxxx
# # Second Data Group for model prediction without "Credit History" and "Foreign Worker" features
#
# df_selected1 = ger_credit_df.drop(columns=['Number of people being liable to provide maintenance for', \
#                                            'Present residence since', 'Job', 'Telephone', 'Credit amount',
#                                            'Number of existing credits at this bank', \
#                                            'Credit history', 'foreign worker'
#                                            ])
# # Second list with names of columns for one-hot encoding of 9 categorical variables for model prediction
# # without "Credit History" and "Foreign Worker" features
# # Final dummy will use 9 categorical variables, and 3 numerical variable and 1 target variable
#
# col_cat_names1 = ["Status of existing checking account", "Purpose", \
#                   "Savings account/bonds", "Present employment since", \
#                   "Personal status and sex", "Other debtors / guarantors", "Property", "Other installment plans",
#                   "Housing"]
#
#
# class Preprocessor:
#     def __init__(self, df_selected1, col_cat_names1):
#         self.df_selected1 = df_selected1
#         self.col_cat_names1 = col_cat_names1
#
#     def one_hot_encode(self):
#         for attr in self.col_cat_names1:
#             self.df_selected1 = self.df_selected1.merge(pd.get_dummies(self.df_selected1[attr], prefix=attr),
#                                                         left_index=True, right_index=True)
#             self.df_selected1.drop(attr, axis=1, inplace=True)
#
#     def replace_value(self):
#         risk = {"Good Risk": 1, "Bad Risk": 0}
#         self.df_selected1["Credit Risk"] = self.df_selected1["Credit Risk"].map(risk)
#
#
# preprocessor = Preprocessor(df_selected1, col_cat_names1)
# preprocessor.one_hot_encode()
# preprocessor.replace_value()
#
# df_dummies1 = preprocessor.df_selected1
#
# #Save as CSV file
# df_dummies1.to_csv("df_dummies1_09_march_2024.csv",index=False)

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

# x_test
#
#
# # XGBoost Model Training - Data Group 2 - No Credit History or Immigrant tag
#
# xgb = XGBClassifier(max_depth=2,                 # Depth of each tree
#                             learning_rate=0.1,            # How much to shrink error in each subsequent training. Trade-off with no. estimators.
#                             n_estimators=50,             # How many trees to use, the more the better, but decrease learning rate if many used.
#                             verbosity=1,                  # If to show more errors or not.
#                             objective='binary:logistic',  # Type of target variable.
#                             booster='gbtree',             # What to boost. Trees in this case.
#                             n_jobs=2,                    # Parallel jobs to run. Set your processor number.
#                             gamma=0.001,                  # Minimum loss reduction required to make a further partition on a leaf node of the tree. (Controls growth!)
#                             subsample=0.632,              # Subsample ratio. Can set lower
#                             colsample_bytree=1,           # Subsample ratio of columns when constructing each tree.
#                             colsample_bylevel=1,          # Subsample ratio of columns when constructing each level. 0.33 is similar to random forest.
#                             colsample_bynode=1,           # Subsample ratio of columns when constructing each split.
#                             reg_alpha=1,                  # Regularizer for first fit. alpha = 1, lambda = 0 is LASSO.
#                             reg_lambda=0,                 # Regularizer for first fit.
#                             scale_pos_weight=1,           # Balancing of positive and negative weights. G / B
#                             base_score=0.5,               # Global bias. Set to average of the target rate.
#                             random_state=0,        # Seed
#                             #missing=None,                 # How are nulls encoded?
#                             #tree_method='gpu_hist',       # How to train the trees?
#                             #gpu_id=0                      # With which GPU?
#                             )
#
#
# xgb.fit(x_train, y_train)
#
# # Prediction
# y_prediction_xgb = xgb.predict(x_test)
#
#
#
# # Save the model to a file
# joblib.dump(xgb, 'xgboost_credit_risk_no_credit_history.pkl')


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
    duration = st.slider('Duration in months', 0, 100, 12)
    installment_rate = st.slider('Installment rate in % of disposable income', 0, 10, 5)
    age = st.slider('Age in years', 18, 1000, 30)

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

        # Interpret the prediction
        predicted_label = 'Your £1000 credit has been approved' if prediction[
                                                                       0] == 1 else 'Your £1000 credit application has been rejected'
        st.subheader("Outcome:")
        st.write(predicted_label)

if __name__ == "__main__":
    main()
