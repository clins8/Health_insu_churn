import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Sets the title of the Streamlit dashboard
st.title("Health Insurance Churn Prediction Dashboard")

# Loads the dataset directly from local path
df = pd.read_csv(r"D:\Insurancechurn\insurance.csv", encoding= 'ISO-8859-1')

# Creates a 'churn' variable where customers with charges above the median are labeled as 1
df['churn'] = df['charges'].apply(lambda x: 1 if x > df['charges'].median() else 0)

# Encodes categorical variables into numerical format using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Displays a sample of the processed data
st.subheader("Sample Data")
st.write(df_encoded.head())

# Visualizes churn distribution using a count plot
st.subheader("Churn Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='churn', data=df_encoded, ax=ax1)
st.pyplot(fig1)

# Prepares the features and target variable, splits into train and test sets
X = df_encoded.drop(['charges', 'churn'], axis=1)
y = df_encoded['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trains a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Displays model performance using classification report and confusion matrix
st.subheader("Model Evaluation")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

fig2, ax2 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax2)
st.pyplot(fig2)
