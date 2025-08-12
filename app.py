"""
Netflix Movie Search API - Improved with Semantic Understanding
Uses Sentence Transformers for semantic search to handle natural language queries
"""
import torch
from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and data
model = None
movies_df = None
movie_embeddings = None

def setup_device():
    """Detect and configure GPU/CPU device"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("No GPU detected, using CPU")
        return 'cpu'

def load_and_prepare_data():
    """Load movie data and prepare for semantic search"""
    global movies_df, model, movie_embeddings
    
    print("Loading movie dataset...")
    # Load main dataset
    movies_df = pd.read_csv('movie_data/movies_metadata.csv', low_memory=False)
    
    # Clean and prepare data
    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
    movies_df['vote_average'] = pd.to_numeric(movies_df['vote_average'], errors='coerce')
    movies_df['vote_count'] = pd.to_numeric(movies_df['vote_count'], errors='coerce')
    
    # Parse genres
    def parse_genres(genres_str):
        try:
            if pd.isna(genres_str):
                return []
            genres_list = eval(genres_str)
            return [genre['name'] for genre in genres_list] if isinstance(genres_list, list) else []
        except:
            return []
    
    movies_df['genres_list'] = movies_df['genres'].apply(parse_genres)
    
    # Filter movies with essential data
    movies_df = movies_df.dropna(subset=['title', 'release_date', 'vote_average'])
    movies_df = movies_df[movies_df['vote_count'] >= 10]  # At least 10 votes
    
    # Load additional data for better context
    try:
        # Load keywords for semantic richness
        keywords_df = pd.read_csv('movie_data/keywords.csv')
        keywords_df['keywords_list'] = keywords_df['keywords'].apply(lambda x: parse_json_field(x))
        
        # Load credits for cast/director info
        credits_df = pd.read_csv('movie_data/credits.csv')
        credits_df['cast_list'] = credits_df['cast'].apply(lambda x: get_top_cast(x, 3))
        credits_df['director'] = credits_df['crew'].apply(get_director)
        
        # Merge additional data
        movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
        keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce')
        credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce')
        
        movies_df = movies_df.merge(keywords_df[['id', 'keywords_list']], on='id', how='left')
        movies_df = movies_df.merge(credits_df[['id', 'cast_list', 'director']], on='id', how='left')
    except:
        print("Additional data files not found, using basic movie info only")
        movies_df['keywords_list'] = [[] for _ in range(len(movies_df))]
        movies_df['cast_list'] = [[] for _ in range(len(movies_df))]
        movies_df['director'] = ''
    
    # Fill NaN values
    movies_df['keywords_list'] = movies_df.get('keywords_list', pd.Series([[] for _ in range(len(movies_df))]))
    movies_df['cast_list'] = movies_df.get('cast_list', pd.Series([[] for _ in range(len(movies_df))]))
    movies_df['director'] = movies_df.get('director', pd.Series(['' for _ in range(len(movies_df))]))
    movies_df['overview'] = movies_df['overview'].fillna('')
    
    print(f"Loaded {len(movies_df)} movies")
    
    # Initialize semantic model
    device = setup_device()
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    if device != 'cpu':
        model = model.to(device)
    
    # Prepare content for embedding
    print("Preparing movie content for semantic embedding...")
    def create_content_text(row):
        """Create rich text representation of movie for embedding"""
        content = []
        
        # Add genres
        if isinstance(row.get('genres_list'), list) and row['genres_list']:
            content.append(' '.join(row['genres_list']))
        
        # Add keywords
        if isinstance(row.get('keywords_list'), list) and row['keywords_list']:
            content.extend(row['keywords_list'][:5])
        
        # Add director
        if row.get('director'):
            content.append(f"directed by {row['director']}")
        
        # Add cast
        if isinstance(row.get('cast_list'), list) and row['cast_list']:
            content.append(f"starring {', '.join(row['cast_list'][:3])}")
        
        # Add overview
        if row.get('overview'):
            content.append(row['overview'])
        
        # Add title for better matching
        content.append(row['title'])
        
        return ' '.join(str(item) for item in content)
    
    movies_df['content_text'] = movies_df.apply(create_content_text, axis=1)
    
    # Generate embeddings
    print("Generating movie embeddings (this may take a moment)...")
    batch_size = 64 if device == 'cuda' else 32
    movie_texts = movies_df['content_text'].tolist()
    movie_embeddings = model.encode(
        movie_texts, 
        batch_size=batch_size,
        show_progress_bar=True,
        device=device
    )
    
    print(f"Generated embeddings for {len(movie_embeddings)} movies")
    print("Movie search API ready!")

def parse_json_field(x):
    """Parse JSON-like fields safely"""
    try:
        if pd.isna(x):
            return []
        json_str = str(x).replace("'", '"')
        data = json.loads(json_str)
        if isinstance(data, list):
            return [item.get('name', '') for item in data if isinstance(item, dict)]
        return []
    except:
        return []

def get_top_cast(cast_str, n=3):
    """Extract top N cast members"""
    try:
        if pd.isna(cast_str):
            return []
        json_str = str(cast_str).replace("'", '"')
        cast_list = json.loads(json_str)
        return [actor['name'] for actor in cast_list[:n]] if isinstance(cast_list, list) else []
    except:
        return []

def get_director(crew_str):
    """Extract director from crew data"""
    try:
        if pd.isna(crew_str):
            return ''
        json_str = str(crew_str).replace("'", '"')
        crew_list = json.loads(json_str)
        if isinstance(crew_list, list):
            for person in crew_list:
                if person.get('job') == 'Director':
                    return person.get('name', '')
        return ''
    except:
        return ''

def semantic_search(query, n_results=10):
    """Perform semantic search for movies based on natural language query"""
    if model is None or movie_embeddings is None:
        return []
    
    # Encode query
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    query_embedding = model.encode([query], device=device)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, movie_embeddings).flatten()
    
    # Get top results
    top_indices = similarities.argsort()[::-1][:n_results]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum similarity threshold
            movie = movies_df.iloc[idx]
            results.append({
                'title': movie['title'],
                'release_date': movie['release_date'].strftime('%Y-%m-%d') if pd.notna(movie['release_date']) else 'Unknown',
                'rating': float(movie['vote_average']) if pd.notna(movie['vote_average']) else 0.0,
                'similarity': float(similarities[idx])  # Include for debugging, can remove in production
            })
    
    return results

@app.route('/search', methods=['GET'])
def search_movies():
    """
    Search endpoint for natural language movie queries
    Example: /search?query=horror movies suitable for teenagers
    """
    query = request.args.get('query')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        # Get semantic search results
        results = semantic_search(query, n_results=20)
        
        # Return top results (excluding similarity score for production)
        clean_results = [
            {
                'title': r['title'],
                'release_date': r['release_date'],
                'rating': r['rating']
            }
            for r in results
        ]
        
        return jsonify({'results': clean_results})
    
    except Exception as e:
        print(f"Error processing query '{query}': {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'movies_loaded': len(movies_df) if movies_df is not None else 0,
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("Initializing Netflix Movie Search API...")
    load_and_prepare_data()
    print("\nAPI is ready! Test with:")
    print('curl --get --data-urlencode "query=horror movies suitable for teenagers" http://localhost:5000/search')
    app.run(debug=True, host='0.0.0.0', port=5000)