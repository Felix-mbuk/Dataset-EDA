# Interactive Dataset Exploration and Machine Learning

This project is an interactive web application built using **Streamlit** that allows users to upload a dataset (CSV format), clean and explore it with various visualizations, and even build machine learning models directly within the app. Whether you're analyzing data trends, cleaning datasets, or making predictions using machine learning, this app provides an intuitive interface for it all.

## Features

1. **Upload and Preview Dataset**  
   - Upload a CSV file and preview the first few rows to inspect the data.
   
2. **Data Cleaning**  
   - Automatically handle missing values by removing rows with null entries.
   - Remove duplicate rows and standardize text columns (remove extra spaces and convert to lowercase).

3. **Dataset Overview**  
   - View a summary of the dataset: number of rows, columns, and column names.
   
4. **Data Visualization**  
   - Visualize numeric and categorical data distributions with histograms and bar plots.
   - Generate a correlation heatmap for numerical columns to analyze relationships.
   - Scatter plots for comparing two numeric columns.

5. **Machine Learning Models**  
   - Build and train either a **Linear Regression** or **Decision Tree** model.
   - Evaluate model performance using **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**.
   - Make predictions based on user-provided input data.

6. **Download Cleaned Data**  
   - After cleaning, download the cleaned dataset as a CSV file.

## Installation

### Clone the Repository

```bash
git clone https://github.com/Felix-mbuk/streamlit-dataset-exploration.git
cd streamlit-dataset-exploration
