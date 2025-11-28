from search_engine import ImageSearchEngine
import os

def debug_search():
    # Initialize engine
    print("Initializing engine...")
    engine = ImageSearchEngine()
    
    # Check if index exists or needs to be created
    # For debugging, we might need to point to a real directory. 
    # Since I don't know the user's directory, I'll ask them to run this script or I'll try to mock it if I had images.
    # But wait, the user already indexed images. The engine should have state if it persists it.
    # Ah, looking at search_engine.py, it DOES NOT persist the index to disk! 
    # It only keeps it in memory. So when the server restarts or if I run a separate script, it's empty.
    
    # This is likely the problem if the user restarted the server, but they said "it shows random images", 
    # which implies it returns SOMETHING. If it was empty, it would return nothing.
    # If it returns random images, it means the embeddings are messed up or the similarity is wrong.
    
    # Let's try to verify the logic with a small test if possible, or just review the code.
    pass

if __name__ == "__main__":
    debug_search()
