import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
abalone_data_path = "abalone.data"
column_names = ["Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight",
                "VisceraWeight", "ShellWeight", "Rings"]

# Read the data into a pandas DataFrame
abalone_df = pd.read_csv(abalone_data_path, names=column_names)

# Drop the "Sex" column since it's categorical and not useful for clustering
abalone_df.drop("Sex", axis=1, inplace=True)

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(abalone_df)

# Choose the number of clusters (k) and initialize the K-Means model
k = 3 
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the model to the scaled data
kmeans.fit(scaled_features)

# Get the cluster assignments for each data point
abalone_df["Cluster"] = kmeans.labels_

# Get the centroids (cluster centers)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

print("Cluster Assignments:")
print(abalone_df[["Length", "Diameter", "Height", "Cluster"]])

print("\nCentroids (Cluster Centers):")
print(pd.DataFrame(centroids, columns=column_names[1:]))  # Exclude "Sex" column

# Create a pairplot to visualize every pair of features
sns.pairplot(abalone_df, hue="Cluster", palette=sns.color_palette("hsv", k))

plt.show()



# python -m venv venv
# pip install numpy scikit-learn
# pip install pandas
#pip install matplotlib seaborn
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# I'll put it here to bypass the security to activate .\venv\Scripts\activate