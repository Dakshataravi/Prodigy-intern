import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data.csv")

# Use relevant features (example)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal K using Elbow Method
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow graph
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Train final model (choose K=5 as example)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster column
data['Cluster'] = clusters

# Save result
data.to_csv("clustered_data.csv", index=False)

# Plot clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.title("Customer Segments")
plt.xlabel("Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()

print("Clustering completed. Check clustered_data.csv")
