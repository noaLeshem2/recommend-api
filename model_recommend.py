import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pymongo import MongoClient
import joblib
from dotenv import load_dotenv
import os

def generate_similarity_matrix():
    # Connect to MongoDB
    load_dotenv()  # take environment variables from .env.
    client = MongoClient(os.getenv('MONGO_CONNECTION_STRING'))
    db = client.test

    # Fetch all items and users
    items = list(db['items'].find({}))
    users = list(db['users'].find({}))

    # Convert to DataFrame for easier manipulation
    items_df = pd.DataFrame(items)
    users_df = pd.DataFrame(users)

    # Convert labels list to strings
    items_df['labels'] = items_df['labels'].apply(' '.join)

    # Use CountVectorizer to convert labels to feature vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(items_df['labels'])

    # Compute similarity matrix based on item features
    item_similarity_matrix = cosine_similarity(X)

    city_similarity = pd.read_csv('similarity_matrix_final.csv', index_col=0)

    # Initialize matrices
    location_scores = np.zeros((len(users_df), len(items_df)))
    size_scores = np.zeros((len(users_df), len(items_df)))
    social_scores = np.zeros((len(users_df), len(items_df)))
    favorites_scores = np.zeros((len(users_df), len(items_df)))

    # For each user, compute scores based on location, size, social and favorites
    for i, user in users_df.iterrows():
        user_city = user['city']
        user_size = user['size']
        following = set(user['following'])
        favorites = set(user['favItems'])

        for j, item in items_df.iterrows():
            item_location = item['itemLocation']
            item_size = item['size']
            item_seller = item['sellerUsername']
            item_id = item['_id']

            if user_city in city_similarity.index and item_location in city_similarity.columns:
                scores = city_similarity.loc[user_city, item_location]
                if isinstance(scores, pd.Series):
                    print(f"Warning: Multiple scores for city pair ({user_city}, {item_location}): {scores}")
                    score = scores.mean()  # or scores.iloc[0] or scores.iloc[-1] to use only the first or last score
                else:
                    if scores >= 0.94:
                        location_scores[i, j] = scores
                        score = 1.5 * scores
                    else:
                        location_scores[i, j] = 0.2  # or any default value you want to set
                        score = scores
                print(f"user_city: {user_city}, item_location: {item_location}, score: {scores}")  # Debugging line

            else:
                location_scores[i, j] = 0.2  # or any default value you want to set

            #location_scores[i, j] = city_similarity.loc[user_city, item_location]

            # Compute size score
            # size_difference = abs(int(user_size) - int(item_size))
            # Compute size score
            size_difference = abs(int(user_size) - int(item_size))
            size_scores[i, j] = 1 / (size_difference / 2 + 1)
            #size_scores[i, j] = 1 - (size_difference / 2) if size_difference <= 2 else 1 / size_difference

            # Compute social score
            # Give high score if user is following the item's seller

            # max you can get is 2.
            social_scores[i, j] += 0.5 * (item_seller in following) # + 0.5 * (item_id in favorites)

            # Compute favorites score
            #print(f"item id is: {item_id}!!!!!!!!!!")
            #print(f"favorites are: {favorites}!!!!!!!!!!")
            item_id_str = str(item_id)

            if item_id_str  in favorites:
                #print("inside favoriets!")
                # Find the index of this item in items dataframe
                fav_index = items_df.index[items_df['_id'] == item_id][0]
                # Get top 5 similar items
                similar_item_indices = item_similarity_matrix[fav_index].argsort()[-6:-1][::-1]
                favorites_scores[i, similar_item_indices] += 0.2

    # Normalize the scores to the same scale

    # similarity_matrix = size_scores + social_scores + favorites_scores + location_scores
    scaler = MinMaxScaler()
    location_scores = scaler.fit_transform(location_scores)
    size_scores = scaler.fit_transform(size_scores)
    social_scores = scaler.fit_transform(social_scores)
    favorites_scores = scaler.fit_transform(favorites_scores)

    # Compute the final similarity matrix by adding the score matrices
    similarity_matrix = size_scores + social_scores + favorites_scores + location_scores

    similarity_matrix = scaler.fit_transform(similarity_matrix)

    # Save the model and similarity matrix for later use
    joblib.dump(vectorizer, 'model.joblib')
    joblib.dump(similarity_matrix, 'similarity_matrix.joblib')
