A note or two on the assignment:
We expect, and suggest, that you spend no more than 4-6 hours getting this done;
We don't want to derail your life, and we know you have other things to do -- feel free to wait for a weekend to get this working;
We'd like you to let us know by end of the next business day, when we can expect the working product;
Do, please, let me know if you have any questions about this assignment, or if there's anything unclear.
In production, we prefer simple, maintainable code.  We'd like to see the same here – We're not looking for cleverness.

Movie Search API
Specification
At Netflix, we maintain a large catalog of movies, shows, games, and more. We continually try to enhance the user experience in all aspects of our service and one is by allowing users to discover content in an easy and intuitive way.
Using whatever libraries or frameworks make sense, improve our experimental Movie Search API. 
The only endpoint this API has is
/search?query=<question>
Where the question is text in natural language. For example, “horror movies suitable for teenagers”. We expect similar questions to produce relevant and very similar results. For example, “spooky movie for teens” should also return similar results. 

Use The Movies Dataset publicly available from Kagle:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
Download and unzip the movies_metadata.csv file. This is the only data source you can use to solve the problem. Note that AI models such as LLMs are most likely trained on these data so you cannot use these.

Our current implementation of this API is the following file app.py


from flask import Flask, request, jsonify 
import pandas as pd
import json

app = Flask(__name__)

# Load the movie data from  https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv
movies_df = pd.read_csv('movies_metadata.csv')

# Preprocess genres: split into a list of strings
movies_df['genres'] = movies_df['genres'].str.split('|')

# Function to extract genre from query text
def extract_genres(query):
    # Simple mapping for this example; you can expand it as needed
    genre_map = {
        'horror': ['horror'],
        'action': ['action'],
        'comedy': ['comedy', 'romance'],  # Example mappings
        'drama': ['drama']
    }
    
    # Extract words that might indicate genres
    words = query.lower().split()
    possible_genres = []
    for word in words:
        if word in genre_map:
            possible_genres.extend(genre_map[word])
    return list(set(possible_genres))  # Return unique genres

# Route to handle the search request
@app.route('/search', methods=['GET'])
def search_movies():
   
    query = request.args.get('query') 
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    target_genres = extract_genres(query)
   
    # Filter movies based on genres
    results = []
    for index, movie in movies_df.iterrows():
        for genre_str in movie['genres']:
            genres = eval(genre_str)
            if any(genre['name'].lower() in [tg.lower() for tg in target_genres] for genre in genres):
                results.append({
                    'title': movie['title'],
                    'release_date': movie['release_date'],
                    'rating': movie['vote_average']
                })
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)

Install the dependencies in a requirements.txt file

fastapi
pandas
flask


pip install -r requirements.txt

To test the application, you can use the following curl command to search for a movie:
export QUERY="A horror movie suitable for teenagers in the 12-15 age range"
curl --get --data-urlencode "query=${QUERY}" http://localhost:5000/search | jq '.results[:3]'
[
  {
    "rating": 5.7,
    "release_date": "1995-12-22",
    "title": "Dracula: Dead and Loving It"
  },
  {
    "rating": 6.9,
    "release_date": "1996-01-19",
    "title": "From Dusk Till Dawn"
  },
  {
    "rating": 6.1,
    "release_date": "1995-09-08",
    "title": "Screamers"
  }
]
export QUERY="A spooky movie suitable for teenagers in the 12-15 age range"
{
  "results": []
}
export QUERY="War commandos"
curl --get --data-urlencode "query=${QUERY}" http://localhost:5000/search | jq
{
  "results": []
}
