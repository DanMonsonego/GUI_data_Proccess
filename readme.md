# Data-Procces-GUI-Project

A sophisticated data preprocessing and modeling application created by a team of adept developers.
This project is an epitome of how various Python libraries can be harmoniously integrated to build a user-friendly GUI for data processing and machine learning tasks.

## 📚 Technology Stack

### Data Preprocessing and Modeling Libraries:
- `sklearn` (for various models and preprocessing tasks)
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `pickle` (for model serialization and deserialization)

### GUI Framework:
- `streamlit` (utilized for creating an interactive web-based GUI)

## 📁 Project Structure

- **GUI.py**:
  - Hosts all GUI components of the project.
  - Entry point of the application.

- **models_preprocess.py**:
  - **import_and_install**: Automatically imports and installs necessary packages.
  - **get_df**: Accepts file path and returns a Pandas DataFrame.
  - **drop_rows**: Drops rows with missing values in the class column.
  - **fill_missing_values**: Fills missing values in the dataset.
  - **Models**: id3, knn, K-means, Naive Bayes, Custom id3, Custom Naive Bayes.
  - **Visualization Functions**: Generate matrices to visualize model results.
  - **save_model_results**: Serializes model results to a binary file using pickle.

## 🚀 Getting Started

1. Ensure all necessary packages are installed.
2. Open terminal or command prompt.
3. Execute the following command: 
   ```bash
   streamlit run GUI.py
