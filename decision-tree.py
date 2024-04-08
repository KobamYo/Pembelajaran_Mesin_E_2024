# Source : https://www.kaggle.com/code/carrie1/decision-tree-classification-using-zoo-animals

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("zoo.data", names=["Animal", "Hair", "Feathers", "Eggs", "Milk", "Airborne",
                                     "Aquatic", "Predator", "Toothed", "Backbone", "Breathes", "Venomous",
                                     "Fins", "Legs", "Tail", "Domestic", "Catsize", "Type"])
df2 = df.merge(df, how='left', left_on='class_type', right_on='Class_Number')

# Visualizations
sns.countplot(df2['Class_Type'], label="Count", order=df2['Class_Type'].value_counts().index)
plt.show()

feature_names = ['Hair', 'Feathers', 'Eggs', 'Milk', 'Airborne', 'Aquatic', 'Predator', 'Toothed',
                 'Backbone', 'Breathes', 'Venomous', 'Fins', 'Legs', 'Tail', 'Domestic']

for f in feature_names:
    g = sns.FacetGrid(df2, col="Class_Type", row=f, hue="Class_Type")
    g.map(plt.hist, "ct")
    g.set(xticklabels=[])
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f)
plt.show()

# Model Training and Evaluation
X = df[feature_names]
y = df['class_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))
