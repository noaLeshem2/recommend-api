import heapq

import numpy as np
from pymongo import MongoClient
from flask import Flask, request
import pandas as pd
import joblib
import schedule
import time
import threading
from model_recommend import generate_similarity_matrix  # Import the function
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Function to update the model
def update_model():
    generate_similarity_matrix()  # Call the function to generate the similarity matrix
    print("Updated the recommendation model.")


# Run once at the beginning
update_model()

# Schedule the job every 30 minutes
schedule.every(30).minutes.do(update_model)


# Function to run the scheduler in a separate thread
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)


# Start the scheduler
scheduler_thread = threading.Thread(target=run_schedule)
scheduler_thread.start()


@app.route('/recommend', methods=['POST'])
def recommend():
    # Load the model and similarity matrix
    vectorizer = joblib.load('model.joblib')
    similarity_matrix = joblib.load('similarity_matrix.joblib')

    # Connect to MongoDB
    load_dotenv()  # take environment variables from .env.
    client = MongoClient(os.getenv('MONGO_CONNECTION_STRING'))
    db = client.test

    # Replace 'your-collection-name' with the actual name of your collection.
    items_collection = db['items']
    users_collection = db['users']

    # Fetch all items and users
    items = list(items_collection.find({}))
    users = list(users_collection.find({}))

    print(f"Number of items: {len(items)}")
    print(f"Number of users: {len(users)}")

    # Convert to DataFrame for easier manipulation
    items_df = pd.DataFrame(items)
    users_df = pd.DataFrame(users)
    print(" in inside recommend")


    user_id = request.json['userId']
    #item_type = request.json['itemType']  # This is new
    # num_recommendations = request.json.get('num', len(items_df))  # Changed to recommend all items

    print(items_df.columns)

    # Filter items by type
    # items_df = items_df[items_df['category'] == item_type]  # Assuming 'type' is the column containing item type

    # Find the row corresponding to this user
    user_row = users_df[users_df['username'] == user_id].index[0]

    # Compute the user-item similarity vector for this user
    user_vector = similarity_matrix[user_row]

    # Get the indices of items sorted by their similarity score
    similar_items = np.argsort(user_vector)[::-1]

    # Get their IDs
    ids = [str(id) for id in items_df.iloc[similar_items]['_id'].tolist()]

    # Compute the user-item similarity vector for each item
    #similarity_vectors = [
    #    similarity_matrix[user_row] * similarity_matrix[:, items_df[items_df['_id'] == item_id].index[0]] for item_id in
    #    items_df['_id'].tolist()]


    # Get the most similar items
    #similar_items = heapq.nlargest(num_recommendations, range(len(similarity_vectors)),
    #                               key=similarity_vectors.__getitem__)

    # Get their IDs
    #ids = items_df.iloc[similar_items]['_id'].tolist()

    return {'ids': ids}


@app.route('/test', methods=['GET'])
def test():
    return "API is working."


if __name__ == "__main__":
    app.run(port=5000, debug=True)

