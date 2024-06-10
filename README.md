# Credit-Fraud-Detection-Analysis
 
A credit card is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash. It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. We have to build a classification model to predict whether a transaction is fraudulent or not.

#import important libraris

1>import numpy as np
2>import pandas as pd
3>import seaborn as sns
4>import matplotlib.pyplot as plt
5>from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
6>from sklearn.linear_model import LogisticRegression
7>from sklearn.ensemble import RandomForestClassifier
8>from sklearn.preprocessing import StandardScaler
9>from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
10>from sklearn.ensemble import IsolationForest

# load the data into pandas dataframe

credit_fraud_data = pd.read_csv('C:/Users/rohan/Downloads/creditcard.csv')
credit_fraud_data

# validate the shape of the data 
credit_fraud_data.shape

# check for datatypes of colums
credit_fraud_data.info()

# lets veriefy missing vallues 
credit_fraud_data.isnull().sum()

# using this code we can see last 9 rows of data 
credit_fraud_data.tail(9)

# Checking the count of the missing values percentage, there are very few missing values there in the dataset
credit_fraud_data.isnull().sum()/len(credit_fraud_data)*100

# The below command will help us understand the total number of columns present in the dataset
len(credit_fraud_data.columns)

# transpose if a nice way of discribing the data ,we just need to do a,T after the dataframe and transpose show you stetistical terms above  as column
credit_fraud_data.describe().T

# lets we check how many columns are in the dataset.
credit_fraud_data.columns

# Check for duplicate records
duplicate_rows = credit_fraud_data.duplicated().sum()
print('Number of duplicate records:', duplicate_rows)

# Display the distribution of legitimate transactions and fraudulent transactions
print(credit_fraud_data['Class'].value_counts())

# get fraud and normal dataset.

fraud = credit_fraud_data[credit_fraud_data["Class"]== 1]
normal = credit_fraud_data[credit_fraud_data["Class"]== 0]
print(fraud.shape,normal.shape)

# Box plots for 'Amount' and 'Time' by Class
plt.figure(figsize=(12, 6))
sns.boxplot(x='Class', y='Amount', data=credit_fraud_data, showfliers=False,)
plt.title('Box Plot of Transaction Amount by Class')
plt.legend(["fraud not detected"])
plt.show()

plt.figure(figsize=(15,8))
sns.barplot(data=credit_fraud_data, y = 'Time',x='Class')
plt.title('Box Plot of Transaction Time by Class')
plt.show()

# Checking for class distribution¶
sns.countplot(x="Class",data=credit_fraud_data)
plt.title('Distribution of Frauds(0: No Fraud || 1: Fraud')

# usig this method we can check Highly imbalanced dataset with 99% of data as not-fraud and only 0.03% of data as fraud
(credit_fraud_data["Class"].value_counts()/284807)*100

# Visualize KDE plot of transaction amount by class
plt.figure(figsize=(12, 6))
sns.kdeplot(credit_fraud_data[credit_fraud_data['Class'] == 0]['Amount'], label='Class 0')
sns.kdeplot(credit_fraud_data[credit_fraud_data['Class'] == 1]['Amount'], label='Class 1')
plt.title('KDE Plot of Transaction Amount by Class')
plt.xlim(0, 2000)  # Limiting x-axis for better readability
plt.show()

ax = sns.histplot(data=credit_fraud_data, x=credit_fraud_data["Time"], kde=True)
ax.set_title("Distribution of Transaction Time")

credit_fraud_data.hist(figsize=(20,20),color = 'pink')
plt.show()

# Explore feature distributions
plt.figure(figsize=(14, 12))
for i in range(1, 29):  # Assuming V1 to V28 are the feature columns
    plt.subplot(7, 4, i)
    sns.histplot(credit_fraud_data[f'V{i}'], bins=30, kde=True,color = 'blue')
    plt.title(f'Distribution of V{i}')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(15,10))
sns.heatmap(credit_fraud_data.corr(), annot= True, fmt='.1f', cmap='coolwarm_r')
plt.show()

# Handling outliers using Isolation Forest for 'Amount' and 'Time'
outlier_detector = IsolationForest(contamination=0.01, random_state=1)
credit_fraud_data['Outlier'] = outlier_detector.fit_predict(credit_fraud_data[['Amount', 'Time']])
credit_fraud_data = credit_fraud_data[credit_fraud_data['Outlier'] == 1].drop(columns='Outlier')




