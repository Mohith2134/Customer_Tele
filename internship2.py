# Telco Customer Churn Prediction

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import time

# 1. Load the dataset
df = pd.read_csv("telco_customer_churn_dataset.csv")
print("Loading the dataset")
start=time.sleep(5)
print("Shape of data:", df.shape)
print(df.head())

sns.barplot(x='gender',y='TotalCharges', data=df)
plt.title("Overall gender TotalCharges")
plt.show()
 
sns.boxenplot(x='Contract',data=df)
plt.title("Contract For the Payments")
plt.show()

churn_counts = df['Churn'].value_counts()
labels = churn_counts.index
sizes = churn_counts.values
colors = ['blue', 'green']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, wedgeprops=dict(width=0.4))
plt.title("Customer Churn Distribution")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# 2. Drop irrelevant or ID column if exists
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# 3. Handle missing values (e.g., TotalCharges can be object with missing values)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.ffill(method='ffill', inplace=True)

# 4. Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 5. Split the dataset
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Train the RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Predictions and evaluation
y_pred = model.predict(X_test)
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Save predictions
X_test['Predicted_Churn'] = y_pred
X_test.to_csv("churn_predictions.csv", index=False)
