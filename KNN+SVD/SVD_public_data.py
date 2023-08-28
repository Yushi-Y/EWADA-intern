from MovieLens import MovieLens
from surprise import SVD, SVDpp

# Load your data or use any existing Surprise dataset
ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

# Train the model using the entire dataset (no need to split into trainSet and testSet)
trainSet = data.build_full_trainset()

# Create an SVD model (Singular Value Decomposition)
# Add hyperparameters to the SVD algorithm - hyperparameters tuned for the whole dataset
n_factors = 50
n_epochs = 20
lr_all = 0.005
model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all)
# Or use the SVD plus plus model which performs slightly better than SVD
# model = SVDpp()

# Train the model
model.fit(trainSet)

# Specify a random test user ID (integer)
testUserID = 38

# Specify the movie ratings or interactions for the test user (dictionary)
# Replace the following dictionary with your custom movie history for the test user
testUserMovieRatings = {
    260: 5,   # Movie with ID 260 (Star Wars) has a rating of 5
    1196: 5,   # Movie with ID 1196 (The Empire Strikes Back) has a rating of 5
    8961: 5    # Movie with ID 8961 (The Incredibles) has a rating of 5
}

# Create a custom test set for the test user
testSet = trainSet.build_anti_testset()
testSet += [(trainSet.to_raw_uid(testUserID), trainSet.to_raw_iid(movieID), rating)
            for movieID, rating in testUserMovieRatings.items()]

# Get user-based recommendations for the test user
user_recommendations = model.test(testSet)

# Sort the recommendations based on predicted ratings in descending order
user_recommendations.sort(key=lambda x: x.est, reverse=True)

# Get the top N movie recommendations for the test user (e.g., N=10)
top_n_recommendations = user_recommendations[:10]

# Print the top N movie recommendations for the test user
for recommendation in top_n_recommendations:
    movieID = recommendation.iid
    predicted_rating = recommendation.est
    print(f"Movie Name: {ml.getMovieName(int(movieID))}, Predicted Rating: {predicted_rating}")