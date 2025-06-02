🧑‍⚕️Health Insurance Churn Prediction Dashboard

This is a fun and beginner-friendly project where I built a simple machine learning dashboard using Streamlit. It uses health insurance data to predict if a customer is likely to churn

-What’s This Project About?

I used a dataset with customer details like age, gender, BMI, number of children, smoking status, and medical charges. I created a new column called churn, where:

* Customers with charges higher than the median are marked as churned (`1`)
* Others are marked as not churned (`0`)

Then I trained a Logistic Regression model to predict this churn.

-What I Did:

* Cleaned and prepped the data
* Used one-hot encoding for `sex`, `smoker`, and `region`
* Trained the model using `LogisticRegression`
* Built a Streamlit dashboard to:

  * Show a sample of the data
  * Visualize churn distribution
  * Display model performance (classification report & confusion matrix)

-What You’ll See

* A sample of the data (after encoding)
* A bar chart showing how many customers churned
* Model performance using classification metrics
* A confusion matrix heatmap

