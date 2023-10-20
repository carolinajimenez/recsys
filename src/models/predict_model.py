"""
Recommendation System - LigthFM Model Inference
"""

import numpy as np
import pandas as pd

from helpers.logger import Logger
from helpers.save_files import read_file_pkl


class LightFMPredictor():
    """
    LigthFM Model Predictor Class
    """

    def __init__(self):
        """
        Constructor method.
        """
        # LOG
        self.inference_log = Logger("LightFMPredictor", show_asctime="dev").get_logger()

        # MODEL & DATASET
        self.lightfm_model = None
        self.lightfm_dataset = None

    def load_predicting_data(self):
        """
        Load predicting data (LigthFM model and dataset)
        """
        save_path = "models"
        self.lightfm_model = read_file_pkl(
            save_path=save_path,
            filename="lightfm_model",
        )
        self.lightfm_dataset = read_file_pkl(
            save_path=save_path,
            filename="lightfm_dataset",
        )

    def sample_recommendation_user(self, user_id, threshold = 0, nrec_items = 10):
        """
        Function to produce user recommendations

        Args: 
            * user_id = user ID for which we need to generate recommendation
            * threshold = value above which the rating is favorable in new interaction matrix
            * nrec_items = Number of output recommendation needed

        Returns:
            * Prints list of items the given user has already bought
            * Prints list of N recommended items  which user hopefully will be interested in
        """
        self.inference_log.info("Get recommendations")

        # Get user-id map according lightfm dataset
        userId = float(user_id)
        user_x = self.lightfm_dataset['user_id_map'][userId]

        # Get list of movies that user has rated with a value greater than 4 (rating)
        ratings = pd.read_csv(f"data/raw/ratings.csv")
        filtered_ratings = ratings[(ratings['userId'] == userId) & (ratings['rating'] > 4.0)]
        # Check if the list is empty and if so, assign an empty list
        if filtered_ratings.empty:
            movie_ids_by_user = []
        else:
            # Get list of movieId for specific user
            movie_ids_by_user = filtered_ratings['movieId'].tolist()       
        # Remove duplicates from list
        excluded_items = list(set(movie_ids_by_user))

        # Get item-ids (movies) available to recommend
        items_availables_to_rec = self.lightfm_dataset['item_id_map']
        values_availables_to_rec = []
        # Filter item idx that can be taken into account
        if len(excluded_items) != 0 and items_availables_to_rec is not None:
            items_availables_to_rec = {
                k: v for k,
                v in items_availables_to_rec.items() if k not in excluded_items}
            values_availables_to_rec = list(items_availables_to_rec.values())

        scores = self.lightfm_model.predict(
            user_ids=user_x,
            item_ids=values_availables_to_rec,
            user_features=self.lightfm_dataset["user_features_matrix"],
            item_features=self.lightfm_dataset["item_features_matrix"],
        )

        # Sort items by predicted scores in descending order
        scores_sorted_indices = np.argsort(-scores)

        # Select the top N recommended items
        self.inference_log.info("Select n-recommend items")
        return_score_list = scores_sorted_indices[:nrec_items]
        return return_score_list

    def recommend_movies_by_genre(self, movies, genre, nrec_items=10):
        """
        Function to recommend movies based on genre

        Args:
            * movies: DataFrame containing movie information, including 'movieId', 'title', and 'genres'
            * genre: Genre for which you want movie recommendations
            * nrec_items: Number of output recommendations needed

        Returns:
            * Returns a DataFrame containing recommended movies of the specified genre
        """
        # Identify the movies of the specified genre
        genre_movies = movies[movies['genres'].str.contains(genre)]
        
        # Extract the 'movieId' values of genre-specific movies
        genre_movie_ids = genre_movies['movieId'].values

        _, n_items = self.lightfm_dataset["interactions"].shape
        scores = self.lightfm_model.predict(
            user_ids=0,
            item_ids=np.arange(n_items),
            user_features=self.lightfm_dataset["user_features_matrix"],
            item_features=self.lightfm_dataset["item_features_matrix"],
        )

        # Sort items by predicted scores in descending order
        scores_sorted_indices = np.argsort(-scores)
        
        # Select the top N recommended items from genre-specific movies
        recommended_movie = []
        for item_index in scores_sorted_indices:
            if movies.iloc[item_index]['movieId'] in genre_movie_ids:
                recommended_movie.append(item_index)
            if len(recommended_movie) == nrec_items:
                break        
        return recommended_movie

    def run(self, user_id, movie_genre, movies):
        """
        Run the LightFM recommendation system to generate recommendations for a user or movies based on genre.

        Args:
            * user_id (int or None): User ID for which recommendations will be generated.
                        Set to None if movie recommendations based on genre are needed.
            * movie_genre (str or None): Genre for which movie recommendations are needed.
                        Set to None if user-specific recommendations are needed.
            * movies (pd.DataFrame): DataFrame containing movie information, including 'movieId', 'title', and 'genres'.

        Returns:
            * rec_list (list): A list of recommended movie IDs.
                        If user_id is provided, it contains movie recommendations for the user.
                        If movie_genre is provided, it contains movie recommendations based on genre.

        Note:
            - If user_id is not None, the method will generate movie recommendations for the specified user.
            - If movie_genre is not None, the method will generate movie recommendations based on the specified genre.
            - The recommendations are determined using the trained LightFM model and the provided dataset.
            - The number of recommendations returned is controlled by the method's logic and parameters.
        """
        self.load_predicting_data()

        rec_list = None
        if user_id is not None:
            rec_list = self.sample_recommendation_user(
                user_id=user_id,
                threshold=4,
                nrec_items=10,
            )
        else:
            rec_list = self.recommend_movies_by_genre(
                movies=movies,
                genre=movie_genre,
            )
        return rec_list

        
