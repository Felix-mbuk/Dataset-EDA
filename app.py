import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Welcome title and description for the app
st.title("Interactive Dataset Exploration and Machine Learning")
st.write("Upload a dataset, clean it, explore various visualizations, and even build machine learning models directly from the app.")

# Step 1: Allow the user to upload a dataset (CSV file)
st.header("1. Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        # Try to load the dataset, with fallback encoding handling
        try:
            data = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            # If UTF-8 fails, fall back to ISO-8859-1 encoding
            data = pd.read_csv(uploaded_file, encoding="ISO-8859-1", encoding_errors="ignore")
        
        # Display a preview of the first few rows of the dataset
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        # Step 2: Clean the data by handling missing values and duplicates
        st.header("2. Data Cleaning")
        cleaning_actions = []  # List to track the cleaning steps done

        # Clean up missing values by dropping rows that contain any nulls
        missing_before = data.isnull().sum().sum()
        data = data.dropna(how="any")  # Remove rows with any missing values
        missing_after = data.isnull().sum().sum()
        if missing_before > missing_after:
            cleaning_actions.append(f"Removed {missing_before} missing values.")

        # Remove duplicate rows from the dataset
        duplicates_before = data.duplicated().sum()
        data = data.drop_duplicates()
        duplicates_after = data.duplicated().sum()
        if duplicates_before > duplicates_after:
            cleaning_actions.append(f"Removed {duplicates_before} duplicate rows.")

        # Standardize text columns by trimming spaces and converting to lowercase
        for col in data.select_dtypes(include="object"):
            data[col] = data[col].str.strip().str.lower()
        cleaning_actions.append("Standardized text columns (removed spaces and converted to lowercase).")

        # Display a summary of the cleaning actions
        if cleaning_actions:
            st.write("### Cleaning Summary")
            for action in cleaning_actions:
                st.write(f"- {action}")
        else:
            st.write("No cleaning actions were needed.")

        # Show the cleaned dataset
        st.write("### Cleaned Dataset Preview")
        st.dataframe(data.head())

        # Step 3: Dataset Overview (Structure, columns, etc.)
        st.header("3. Dataset Overview")
        st.write(f"**Number of Rows:** {data.shape[0]}")
        st.write(f"**Number of Columns:** {data.shape[1]}")
        st.write(f"**Columns Names:** {', '.join(data.columns)}")

        # Step 4: Visualizations (Understand data trends)
        st.header("4. Visualize Your Data")

        # Identify numeric and categorical columns
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = data.select_dtypes(include=["object"]).columns

        # Visualize the distribution of a numeric column
        if numeric_cols.any():
            st.subheader("Numeric Data Distribution")
            selected_num_col = st.selectbox("Select a numeric column to visualize:", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(data[selected_num_col], bins=20, kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_num_col}")
            st.pyplot(fig)

        # Visualize the distribution of a categorical column
        if categorical_cols.any():
            st.subheader("Categorical Data Distribution")
            selected_cat_col = st.selectbox("Select a categorical column to visualize:", categorical_cols)
            fig, ax = plt.subplots()
            data[selected_cat_col].value_counts().plot(kind="bar", ax=ax, color="orange", edgecolor="black")
            ax.set_title(f"Distribution of {selected_cat_col}")
            st.pyplot(fig)

        # Show the correlation heatmap for numeric columns
        if numeric_cols.any():
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

        # Scatter plot for any two numeric columns
        if len(numeric_cols) > 1:
            st.subheader("Scatter Plot")
            x_col = st.selectbox("Select X-axis:", numeric_cols)
            y_col = st.selectbox("Select Y-axis:", numeric_cols)
            fig, ax = plt.subplots()
            sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax)
            ax.set_title(f"{x_col} vs {y_col}")
            st.pyplot(fig)

        # Step 5: Build a Machine Learning Model
        st.header("5. Build a Machine Learning Model")

        # Select target and feature columns for modeling
        st.write("**Choose the target (dependent) and feature (independent) variables for model training**")
        target_column = st.selectbox("Select the target column:", data.columns)
        feature_columns = st.multiselect("Select feature columns:", [col for col in data.columns if col != target_column])

        if feature_columns:
            # Split the data into training and testing sets
            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardize the feature values
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Select the model (Linear Regression or Decision Tree)
            model_type = st.selectbox("Select a model type:", ["Linear Regression", "Decision Tree"])

            # Build and train the selected model
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = DecisionTreeRegressor()

            model.fit(X_train_scaled, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test_scaled)

            # Show evaluation metrics (error values)
            st.write("### Model Evaluation")
            st.write(f"**Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred)}")
            st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred)}")

            # Predict for user-provided input
            st.write("### Make a Prediction")
            sample_input = {}
            for feature in feature_columns:
                sample_input[feature] = st.number_input(f"Enter value for {feature}")

            if st.button("Predict"):
                sample_input_df = pd.DataFrame([sample_input])
                sample_input_scaled = scaler.transform(sample_input_df)
                prediction = model.predict(sample_input_scaled)
                st.write(f"Predicted Value: {prediction[0]}")

        # Step 6: Option to Download the Cleaned Data
        st.header("6. Download Cleaned Data")
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Cleaned Dataset", csv, "cleaned_dataset.csv", "text/csv")

    except Exception as e:
        # If an error occurs, show the error message
        st.error(f"An error occurred: {e}")
else:
    st.info("Upload a dataset to start exploring!")
