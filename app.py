import streamlit as st
import pandas as pd 
import numpy as np
import os 
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Importing necessary libraries for the Streamlit app

# Check if a previously uploaded dataset exists
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)  # Load the dataset if it exists
  

def preproccessing(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Identify numerical and categorical columns
    numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns

    # Scale numerical columns
    if len(numerical_cols) > 0:
        scaler = StandardScaler()  # Initialize the scaler
        df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])  # Scale numerical data
    
    # Encode categorical columns
    if len(categorical_cols) > 0:
        le = LabelEncoder()  # Initialize the label encoder
        for col in categorical_cols:
            df_copy[col] = le.fit_transform(df_copy[col])  # Encode categorical data
    
    return df_copy  # Return the preprocessed DataFrame
    

# Sidebar for navigation and branding
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")  # Add logo
    st.title("Welcome to the autoML project!")  # Add title
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])  # Navigation options
    st.info("This project helps you build a model for your data without using a single line of code!")  # Info message

# Upload page functionality
if choice == "Upload":
    st.title("Upload your file for modelling")
    file = st.file_uploader("Upload your file here")  # File uploader widget
    if file:
        df = pd.read_csv(file, index_col=None)  # Read the uploaded file
        df.to_csv("sourcedata.csv", index=None)  # Save the uploaded file
        st.dataframe(df)  # Display the uploaded data

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)
    
# Modelling page functionality
if choice == "Modelling":
    st.title("Select Problem Type for modelling")
    problem_type = st.radio("", ['Classification', 'Regression', 'Clustering'])  # Choose problem type
    
    # Classification modeling
    if problem_type == "Classification":
        from pycaret.classification import setup, compare_models, save_model, pull
        chosen_target = st.selectbox('Choose the Target Column', df.columns)  # Select target column
        if st.button('Run Classification Model'):
            df = preproccessing(df)  # Preprocess the data
            if df[chosen_target].dtype != 'float64':
                df[chosen_target] = df[chosen_target].astype(float)  # Ensure target column is float
            # Setup and modeling
            setup(df, target=chosen_target)  # Initialize PyCaret setup
            setup_df = pull()  # Get setup data
            st.dataframe(setup_df)  # Display setup data
            
            # Run model comparison
            best_model = compare_models()  # Find the best model
            compare_df = pull()  # Get comparison data
            st.dataframe(compare_df)  # Display comparison data
            best_model_name = compare_df.iloc[0]['Model']  # Model name column
            best_model_accuracy = compare_df.iloc[0]['Accuracy']  # Accuracy metric
        
            # Display results and save model
            st.success(f"The best model is {best_model_name} with an Accuracy score of {best_model_accuracy:.4f}.")
            save_model(best_model, 'best_model')  # Save the best model
            st.success("The best Classification model is saved as 'best_model.pkl'.")

    # Regression modeling
    if problem_type == "Regression":
        from pycaret.regression import setup, compare_models, save_model, pull
        chosen_target = st.selectbox('Choose the Target Column', df.columns)  # Select target column
        if st.button('Run Regression Model'):
            df = preproccessing(df)  # Preprocess the data
            if df[chosen_target].dtype != 'float64':
                df[chosen_target] = df[chosen_target].astype(float)  # Ensure target column is float
            # Setup and modeling
            setup(df, target=chosen_target)  # Initialize PyCaret setup
            setup_df = pull()  # Get setup data
            st.dataframe(setup_df)  # Display setup data
            
            # Run model comparison
            best_model = compare_models()  # Find the best model
            compare_df = pull()  # Get comparison data
            st.dataframe(compare_df)  # Display comparison data
            best_model_name = compare_df.iloc[0]['Model']  # Model name column
            best_model_accuracy = compare_df.iloc[0]['MAE']  # MAE metric
        
            # Display results and save model
            st.success(f"The best model is {best_model_name} with an MAE score of {best_model_accuracy:.4f}.")
            save_model(best_model, 'best_model')  # Save the best model
            # st.success("The best regression model is saved as 'best_model.pkl'.")

    # Clustering modeling
    if problem_type == "Clustering":
        from pycaret.clustering import setup, create_model, assign_model, save_model, pull, evaluate_model
        from sklearn.metrics import silhouette_score
        clustering_models = ['kmeans', 'dbscan', 'hclust', 'meanshift']  # List of clustering models
        if st.button("Run Clustering Model"):
            df = preproccessing(df)  # Preprocess the data
            setup(df)  # Initialize PyCaret setup
            comparison_results = []  # List to store results

            # Test each clustering model
            for model_name in clustering_models:
                try:
                    model = create_model(model_name)  # Create clustering model
                    clustered_df = assign_model(model)  # Assign clusters
                    labels = clustered_df['Cluster']  # Get cluster labels
                    if len(np.unique(labels)) > 1:  # Check if more than one cluster exists
                        score = silhouette_score(df, labels)  # Calculate silhouette score
                        comparison_results.append({'Model': model_name, 'Silhouette Score': score})
                except Exception as e:
                    st.error(f"Error with model {model_name}: {e}")

            # Display results and save best model
            if comparison_results:
                compare_df = pd.DataFrame(comparison_results).sort_values(by='Silhouette Score', ascending=False)
                st.write("Comparison of Clustering Models:")
                st.dataframe(compare_df)  # Display comparison data
                best_model_name = compare_df.iloc[0]['Model']  # Best model name
                st.success(f"Best model is {best_model_name} with Silhouette Score: {compare_df.iloc[0]['Silhouette Score']:.4f}")
                best_model = create_model(best_model_name)  # Create the best model
                save_model(best_model, 'best_clustering_model')  # Save the best model
                # st.success("Best clustering model saved as 'best_clustering_model.pkl'")
            else:
                st.warning("No valid clustering models were found. Please check your data.")

# Download page functionality
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")  # Provide download button for the model
