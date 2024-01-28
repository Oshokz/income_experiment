import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


##File Uploads in Streamlit
st.subheader("File Upload and Display")
uploaded_file = st.file_uploader("Upload a File", type = ["pdf", 
                                                          "png", "jpg", "jpeg"])
# check if the file is properly uploaded
if uploaded_file is not None:
    st.write("Filename: ", uploaded_file.name) 
    st.write("File type: ", uploaded_file.type)
    st.image (uploaded_file)


# Streamlit Forms and Button
st.subheader("Streamlit Form and Button")
# using context manager for form - with
with st.form("User Input Form"):
    #Add the text input
    name = st.text_input("Enter your full name")
    email = st.text_input("Enter your email")
    department = st.selectbox("Select your department", ["HR", "Finance", "Engineering", "Marketing"])
    age = st.slider("Select your age", min_value=18, max_value=90, value=25)
    submit_button = st.form_submit_button("Submit")

# if the user clicks the button
if submit_button: 
    #display user input
    st.write("Name:", name)
    st.write("Email:", email)
    st.write("Department:", department)
    st.write("Age:", age)


# case study : training a model using pycaret
# Data = inbuilt pycaret income data
import pandas as pd
from pycaret.datasets import get_data
from pycaret.classification import setup, compare_models, finalize_model
from pycaret.classification import automl, save_model

#get the data
df = get_data("income", save_copy = True, verbose=False)

def clean_marital_status(status):
    married_list= ["Married-civ-spouse", "Married-AF-spouse"]
    if status in married_list:
        return "Married"
    else:
        return "Not married"

df["marital-status"] = df["marital-status"].apply(lambda x: clean_marital_status(x))
#rename the target column
df.rename(columns = {'income >50K': 'income'}, inplace = True)
df.columns
# Prepare the data
s = setup(data = df, target = "income", fix_imbalance=True, 
          remove_outliers =True, verbose=False)

#save the transformed data
transformed_df = s.dataset_transformed
transformed_df.to_csv("transformed_income.csv", index=False)

#train different models on the data
top_models = compare_models(n_select=3, sort = "F1", verbose=False)

# select the best model
best_model = automl(optimize = 'F1')

# save the best model
save_model(best_model, 'best_model', verbose=False)
