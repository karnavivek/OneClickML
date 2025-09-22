import time
import streamlit as st
import pandas as pd
from main import ModelTraining
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#run this (python3 -m streamlit run st_main.py) or (streamlit run st_main.py)

# st.title("OneClickML")


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.title("🧪 OneClickML")
st.write("Upload your dataset, tweak some hyperparameters and get an overall suggestion on which ML Model you should go with!")

# --- Step 1: Upload Data ---
st.header("1. Upload Your Data")
X_file = st.file_uploader("Upload Input Features (X)", type='csv')
y_file = st.file_uploader("Upload Target Variable (y)", type='csv')

# --- Step 2: Set Parameters ---
st.header("2. Set Splitting Parameters")
choose_test_size = st.slider(
    "Test Set Size",
    min_value=0.1,
    max_value=0.5,
    value=0.2, # A default value
    step=0.05
)
st.write(f"You selected a test size of: **{choose_test_size}**")

running_ocml = st.button("Run OneClickML")
if running_ocml:
    # --- Step 3: Process and Split ---
    if X_file is not None and y_file is not None:
        # Read the uploaded files into pandas DataFrames
        X = pd.read_csv(X_file)
        y = pd.read_csv(y_file)

        st.header("3. Data Preview")
        st.write("Input Features (X) Head:")
        st.dataframe(X.head())

        st.write("Target Variable (y) Head:")
        st.dataframe(y.head())

        # Perform the train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=choose_test_size,
            random_state=42 # for reproducibility
        )

        st.header("4. Split Results")
        st.write("Data has been split successfully! Here are the shapes of the resulting sets:")
        st.markdown(f"""
        - **X_train shape:** `{X_train.shape}`
        - **X_test shape:** `{X_test.shape}`
        - **y_train shape:** `{y_train.shape}`
        - **y_test shape:** `{y_test.shape}`
        """)

        st.header("5. Model Training")
        #for running ALL models in a single iteration:
        model_list = ['linear','DT','MLP','GBM','RF','SVM']

        #for running single model in a single iteration
        # model_list = ['linear']
        
            
        seed = 42

        all_results = []
        with st.spinner("Wait for it...", show_time=False):
            for alg in model_list:
                model = ModelTraining()
                perf = model.run_models(X_train, y_train, alg, seed=seed, X_test=X_test, y_test=y_test, binary=False)
                all_results.append(perf)

            
                time.sleep(1)

            results_df = pd.DataFrame(all_results)
            st.success("Done!")

            st.write("\nModel Performance Summary:\n")
            st.dataframe(results_df)
            
            if results_df['Model'][0]=='Linear Regression':
                st.header(f'\nOneClickML Recommeds "{results_df.loc[results_df['MSE Score'].idxmin(), 'Model']}" Model with Minimum score: {min(results_df['MSE Score'])}')
            else:
                st.header(f'\nOneClickML Recommeds "{results_df.loc[results_df['ROC Score'].idxmin(), 'Model']}" Model with Minimum score: {min(results_df['ROC Score'])}')


    else:
        st.warning("☝️ Please upload both CSV files to proceed.")


