"""
Test script for Netflix Movie Search API
Tests the failing queries from the assignment README
"""

import requests
import json

def test_search_api(query, base_url="http://localhost:5000"):
    """Test a search query against the API"""
    try:
        response = requests.get(f"{base_url}/search", params={"query": query})
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        else:
            print(f"Error: Status code {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to API. Make sure the Flask app is running.")
        return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def main():
    # Test queries from the Netflix assignment README
    test_queries = [
        "A horror movie suitable for teenagers in the 12-15 age range",
        "A spooky movie suitable for teenagers in the 12-15 age range",
        "War commandos",
        "action movies",
        "romantic comedy"
    ]
    
    print("Netflix Movie Search API Test")
    print("=" * 60)
    
    # Test health endpoint first
    try:
        health = requests.get("http://localhost:5000/health")
        if health.status_code == 200:
            health_data = health.json()
            print(f"API Status: {health_data.get('status')}")
            print(f"Movies loaded: {health_data.get('movies_loaded')}")
            print(f"Model loaded: {health_data.get('model_loaded')}")
            print("=" * 60)
        else:
            print("Warning: Health check failed")
    except:
        print("Error: Cannot connect to API. Make sure it's running with:")
        print("python app.py")
        return
    
    # Test each query
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        results = test_search_api(query)
        
        if results:
            print(f"Found {len(results)} results:")
            # Show first 3 results
            for i, movie in enumerate(results[:3], 1):
                print(f"  {i}. {movie['title']} ({movie['release_date']}) - Rating: {movie['rating']}")
            
            if len(results) > 3:
                print(f"  ... and {len(results) - 3} more results")
        else:
            print("  No results found")
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("-" * 40)
    
    # Check if failing queries now work
    failing_queries = [
        "A spooky movie suitable for teenagers in the 12-15 age range",
        "War commandos"
    ]
    
    for query in failing_queries:
        results = test_search_api(query)
        status = "✅ FIXED" if results else "❌ STILL FAILING"
        print(f"{status}: '{query}' - {len(results)} results")

if __name__ == "__main__":
    main()