import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset from a CSV file.
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.dropna(subset=['keywords'], inplace=True)
    return df

# Build a TF-IDF matrix for a list of descriptions.
# Returns the fitted TfidfVectorizer and the resulting matrix.
def build_tfidf_matrix(descriptions):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return vectorizer, tfidf_matrix

# Given a user query, compute the similarity with each item's description and return the top_n similar items.
def get_top_n_similar_items(query: str, 
                            vectorizer: TfidfVectorizer, 
                            tfidf_matrix, 
                            df: pd.DataFrame, 
                            top_n: int = 5):
    # Transform the query into the same TF-IDF space
    query_vec = vectorizer.transform([query])
    
    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get the top_n item indices
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    
    # Build a result list with title, similarity score
    results = []
    for idx in top_indices:
        results.append({
            'title': df.iloc[idx]['title'],
            'similarity': round(float(similarity_scores[idx]), 3)  # optional rounding
        })
    return results

def main():  
    if len(sys.argv) < 2:
        print("Usage: python recommend.py 'Your description of preferences'")
        sys.exit(1)
    
    user_query = sys.argv[1]
    
    # Load data
    data_path = "data/movies.csv"
    df = load_data(data_path)
    
    # Build TF-IDF matrix
    vectorizer, tfidf_matrix = build_tfidf_matrix(df['keywords'])
    
    # Get top similar items
    recommendations = get_top_n_similar_items(user_query, vectorizer, tfidf_matrix, df, top_n=5)
    
    # Print recommendations
    print(f"\nUser Query: {user_query}")
    print("Top Recommendations:")
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. {rec['title']} (Similarity: {rec['similarity']})")

if __name__ == "__main__":
    main()
