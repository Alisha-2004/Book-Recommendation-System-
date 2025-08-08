from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model & clustered data
model = pickle.load(open('kmeans_model.pkl', 'rb'))
books_df = pd.read_csv('books_clustered.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        rating = float(request.form['rating'])
        reviews = int(request.form['reviews'])
        pages = int(request.form['pages'])
    except ValueError:
        return "Invalid input! Please enter numeric values."

    # Predict cluster
    features = pd.DataFrame([[rating, reviews, pages]], 
                             columns=['average_rating', 'ratings_count', 'num_pages'])
    cluster = model.predict(features)[0]

    # Get books from the same cluster
    cluster_books = books_df[books_df['Cluster'] == cluster]

    # Sample up to 5 books
    recommendations = cluster_books.sample(min(len(cluster_books), 5))

    return render_template('result.html', books=recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
