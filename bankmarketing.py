from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Fetch dataset
bank_marketing = fetch_ucirepo(id=222)


# Convert data to pandas DataFrame
X = bank_marketing.data.features 
y = bank_marketing.data.targets 


# Data Exploration
# =============================================================================
# print(df.columns)
# print(df.info())
# print(df.describe())
# 
# =============================================================================
# Encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True).astype(int)
y_encoded = pd.get_dummies(y, drop_first=True).astype(int)

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Model Training
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Model Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1) #handles zero division error
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('prediction')
print(y_pred[:10])
print('actual')
print(y[:10])
print("Accuracy:", accuracy)
print("Precision:", precision)#proportion of positive: true pos + false pos
print("Recall:", recall) #proportion of positive: true pos + false neg = high false negative
print("F1 Score:", f1) #f1 = 2* (precision * recall)/precision + recall - 

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, class_names=['No', 'Yes'])
plt.show()
#Gini = impurity - probability of a node being incorrectly classified 
#num samples that reach node 
#value = classnames = 31970 no, 4198 yes
#class = majority rule 









