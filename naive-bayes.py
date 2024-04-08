# Source : https://github.com/hkordbacheh/Applied-Machine-Learning-3/blob/master/Zoo%20Animal%20Classification%20using%20Naive%20Bayes.ipynb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

# Load data
csv_filename = "zoo.data"
df = pd.read_csv(csv_filename, names=["Animal", "Hair", "Feathers", "Eggs", "Milk", "Airborne",
                                      "Aquatic", "Predator", "Toothed", "Backbone", "Breathes", "Venomous",
                                      "Fins", "Legs", "Tail", "Domestic", "Catsize", "Type"])

# Convert animal labels to numbers
df['Animal'] = pd.factorize(df['Animal'])[0]

# Convert Legs to categorical
df['Legs'] = df['Legs'].astype(str)

# Get binarized 'Legs' columns
legs_dummies = pd.get_dummies(df['Legs'], prefix='Leg')
df = pd.concat([df, legs_dummies], axis=1)
df.drop('Legs', axis=1, inplace=True)  # Drop the original 'Legs' column

# Define features and target
features = list(df.columns[1:-1])
X = df[features]
y = df['Type']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Build Extra Trees Classifier for feature importance
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(14, 10))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# Plot the top 5 feature importances of the forest
best_features = [features[i] for i in indices[:5]]
plt.figure(figsize=(8, 6))
plt.title("Top 5 Feature Importances")
plt.bar(best_features, importances[indices][:5], color="r", yerr=std[indices][:5], align="center")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
