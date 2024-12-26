import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Heart Disease Classification")

st.sidebar.title("Navigation")
view_option = st.sidebar.radio(
    "Select View:",
    ("Show Raw Data", "Show Normalized Data", "Show Model Results")
)

split_option = st.sidebar.radio(
    "Select Data Split Ratio:",
    ("70:30", "80:20", "90:10")
)

# Load Dataset
dataset_path = "heart.csv" 
df = pd.read_csv(dataset_path)

if view_option == "Show Raw Data":
    st.subheader("Dataset Mentah")
    st.dataframe(df.head(15))

X = df.drop(columns='target')
y = df['target']

# Normalize Data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Determine test size based on selected ratio
test_size = 0.3 if split_option == "70:30" else 0.2 if split_option == "80:20" else 0.1

# Split Data
X_train_norm, X_test_norm, y_train, y_test = train_test_split(
    X_normalized, y, test_size=test_size, random_state=42, stratify=y
)

if view_option == "Show Normalized Data":
    st.subheader("Data Sebelum Normalization")
    st.dataframe(X.head(15))

    st.subheader("Data Setelah Normalization")
    st.dataframe(pd.DataFrame(X_normalized, columns=X.columns).head(10))

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression()
}

accuracy_scores = {}
reports = {}
confusion_matrices = {}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)

    accuracy_scores[name] = accuracy_score(y_test, y_pred)
    reports[name] = classification_report(y_test, y_pred, output_dict=True)
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

if view_option == "Show Model Results":
    st.subheader(f"Data Split Ratio: {split_option}")

    for name in models.keys():
        st.subheader(f"{name} - Evaluation")

        st.write(f"**Accuracy**: {accuracy_scores[name]:.2f}")

        st.write("**Classification Report**:")
        report_df = pd.DataFrame(reports[name]).transpose()
        st.dataframe(report_df)

        st.write("**Confusion Matrix**:")
        cm = confusion_matrices[name]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'], ax=ax)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)

    st.subheader("Accuracy Comparison")
    accuracy_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Model', 'Accuracy'])
    st.dataframe(accuracy_df)

    st.bar_chart(accuracy_df.set_index('Model'))
