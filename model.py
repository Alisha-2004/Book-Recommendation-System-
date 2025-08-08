import pandas as pd
from sklearn.cluster import KMeans
import pickle

# Load dataset
df = pd.read_csv('books.csv')

# Selecting features for clustering
X = df[['average_rating', 'ratings_count', 'num_pages']]

# KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Save model and data
pickle.dump(kmeans, open('kmeans_model.pkl', 'wb'))
df.to_csv('books_clustered.csv', index=False)
print("Model trained & saved successfully!")
