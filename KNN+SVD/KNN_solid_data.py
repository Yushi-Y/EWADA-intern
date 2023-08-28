# Example movie name to ID mapping dictionary
movie_name_to_id = {
    'Toy Story (1995)': 1,
    'Inception (2010)': 2,
    'The Hunger Games (2012)': 3,
    'Ice Age 4: Continental Drift (2012)': 4,
    'Gone Girl (2014)': 5,
    'Harry Potter and the Deathly Hallows: Part 1 (2010)': 6,
    'Winnie the Pooh (2011)': 7,
    'Frozen (2013)': 8
}

from surprise import Dataset, Reader, KNNBasic

# Sample movie ratings data for 10 users
# Replace this data with your actual movie ratings data for all users
movie_ratings = [
    ('User1', 'Toy Story (1995)', 5),
    ('User1', 'Inception (2010)', 5),
    ('User2', 'Toy Story (1995)', 5),
    ('User2', 'The Hunger Games (2012)', 5),
    ('User2', 'Ice Age 4: Continental Drift (2012)', 5),
    ('User2', 'Inception (2010)', 5),
    ('User2', 'Gone Girl (2014)', 5),
    ('User3', 'Inception (2010)', 5),
    ('User3', 'Harry Potter and the Deathly Hallows: Part 1 (2010)', 5),
    ('User3', 'Winnie the Pooh (2011)', 5),
    ('User3', 'Frozen (2013)', 5),
]

# Create a Surprise Reader to specify the rating scale
reader = Reader(rating_scale=(1, 5))

# Convert the movie ratings list into a DataFrame
import pandas as pd
movie_ratings_df = pd.DataFrame(movie_ratings, columns=['userID', 'movieName', 'rating'])

# Load the movie ratings data from the DataFrame into a Surprise Dataset
data = Dataset.load_from_df(movie_ratings_df, reader)

# Build the full training set
trainSet = data.build_full_trainset()

# Create a KNNBasic model (user-based collaborative filtering)
model = KNNBasic(k=5, sim_options={'user_based': True})

# Train the model
model.fit(trainSet)

# Target user's ID (raw user ID)
target_user = 'User1'

# Convert raw user ID to inner user ID
target_user_innerID = trainSet.to_inner_uid(target_user)

# Get the list of movies rated by the target user
movies_rated_by_target_user = trainSet.ur[target_user_innerID]

# Find movies not rated by the target user
movies_to_recommend = [movieID for movieID in trainSet.all_items() if movieID not in [movieID for movieID, _ in movies_rated_by_target_user]]

# Recommend movies to the target user
recommendations = []
for movieID in movies_to_recommend:
    movie_name = [movie_name for movie_name, movie_id in movie_name_to_id.items() if movie_id == movieID]
    # print(movie_name)
    if movie_name:
        predicted_rating = model.predict(target_user_innerID, movieID).est
        recommendations.append((movie_name, predicted_rating))

# Sort the recommendations based on predicted ratings in descending order
recommendations.sort(key=lambda x: x[1], reverse=True)

# Print the movie recommendations for the target user
print(f"Recommended movies for {target_user} based on data from all users:")
for movie_name, predicted_rating in recommendations:
    print(f"Movie Name: {movie_name}, Predicted Rating: {predicted_rating}")