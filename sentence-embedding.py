import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import os

using_sentence_transformer = True
# model =  SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
if using_sentence_transformer:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to('cuda:7')
else:   
    from InstructorEmbedding import INSTRUCTOR
    model = INSTRUCTOR('hkunlp/instructor-xl').to('cuda:7')
    
# Function to generate hash
def generate_hash(episode_id):
    return hashlib.md5(episode_id.encode()).hexdigest()

# Function to generate embeddings and store them in a dictionary
def generate_embeddings(episode_list):
    # Initialize the model
    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    embeddings = {}
    for index, episode in enumerate(episode_list):
        try:
            episode_id = episode['episode_id']
            goal = episode['goal']
            
            # Generate embedding
            embedding = model.encode(goal).tolist()  # Convert to list for JSON serialization
            
            # Generate hash of the episode_id
            episode_hash = generate_hash(episode_id)
            
            # Store embedding with hash as the key
            embeddings[episode_hash] = {
                'episode_id': episode_id,
                'goal': goal,
                'embedding': embedding
            }
        except Exception as exc:
                print(exc)
                continue
    print(index)
    return embeddings

# Read episode list from a file
def read_episode_list(file_path):
    with open(file_path, 'r') as file:
        episode_list = json.load(file)
    return episode_list

# Write embeddings to a file
def write_embeddings(file_path, embeddings):
    with open(file_path, 'w') as file:
        json.dump(embeddings, file)

# Main process
def main_generate_sentence_embedding():
    episode_list_path = 'episode_list.json'
    episode_embedding_path = 'episode_embedding.json'
    
    # Read episode list
    episode_list = read_episode_list(episode_list_path)
    
    # Generate embeddings
    embeddings = generate_embeddings(episode_list)
    
    # Write embeddings to a file
    write_embeddings(episode_embedding_path, embeddings)

# Function to read all JSON files from a directory
def read_all_json_files_from_directory(directory_path):
    episode_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                episode_list.extend(data)  # Combine all data into a single list
    return episode_list

# Main function to generate sentence embeddings from files in a directory
def main_generate_sentence_embedding(directory_path):
    episode_embedding_path = 'episode_embedding.json'
    
    # Read episode list from all JSON files in the directory
    episode_list = read_all_json_files_from_directory(directory_path)
    
    # Generate embeddings
    embeddings = generate_embeddings(episode_list)
    
    # Write embeddings to a file
    write_embeddings(episode_embedding_path, embeddings)





# Load embeddings from a file
def load_embeddings(file_path):
    with open(file_path, 'r') as file:
        embeddings = json.load(file)
    return embeddings

# Function to find the top K similar episodes
def _find_top_k_similar(task, embeddings, model, k=5):
    task_embedding = model.encode(task).reshape(1, -1)
    similarities = []
    for episode_hash, data in embeddings.items():
        episode_embedding = np.array(data['embedding']).reshape(1, -1)
        similarity = cosine_similarity(task_embedding, episode_embedding)[0][0]
        similarities.append((data['episode_id'], data['goal'], similarity))
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:k]

# Function to find the top K similar episodes for a given task
def get_top_k_similar_episodes(task, k=5):
    episode_embedding_path = 'episode_embedding.json'   
    # Load embeddings
    embeddings = load_embeddings(episode_embedding_path)   
    # # Initialize model
    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')   
    # Find top K similar episodes
    top_k = _find_top_k_similar(task, embeddings, model, k)  
    for i, (episode_id, goal, similarity) in enumerate(top_k):
        print(f"Top {i + 1}:")
        print(f"Episode ID: {episode_id}")
        print(f"Goal: {goal}")
        print(f"Similarity: {similarity}")
        print()
    return top_k

# Run the main process
if __name__ == "__main__":
    # directory_path =  'dataset/caption_json'
    # main_generate_sentence_embedding(directory_path)
    get_top_k_similar_episodes('Clear the cart on target.com. Add logitech g pro to the cart on target.com',10)