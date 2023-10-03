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
        self.lightfm_model = read_file_pkl(
            save_path="recsys/models",
            filename="lightfm_model",
        )
        self.lightfm_dataset = read_file_pkl(
            save_path="recsys/models",
            filename="lightfm_dataset",
        )

    def sample_recommendation_user(self, model, interactions, user_id, 
                               user_dict, item_dict, threshold = 0, nrec_items = 10, show = True):
        """
        Function to produce user recommendations

        Args: 
            * model = Trained matrix factorization model
            * interactions = dataset used for training the model
            * user_id = user ID for which we need to generate recommendation
            * user_dict = Dictionary type input containing interaction_index as key and user_id as value
            * item_dict = Dictionary type input containing item_id as key and item_name as value
            * threshold = value above which the rating is favorable in new interaction matrix
            * nrec_items = Number of output recommendation needed

        Returns:
            * Prints list of items the given user has already bought
            * Prints list of N recommended items  which user hopefully will be interested in
        """
        self.inference_log.info("Get recommendations")
        _, n_items = interactions.shape
        user_x = user_dict[float(user_id)]
        scores = model.predict(
            user_ids=user_x,
            item_ids=np.arange(n_items),
            user_features=self.lightfm_dataset["user_features_matrix"],
            item_features=self.lightfm_dataset["item_features_matrix"],
        )
        print("scores", scores)

        # Create a mask to filter out items the user has interacted with above the threshold
        known_items_mask = interactions.getrow(user_x).toarray().flatten() > threshold
        print("known_items_mask", known_items_mask)
        
        # Set scores for known items to a very low value (e.g., -1e9) to filter them out
        scores[known_items_mask] = -1e9
        print("scores", scores)

        # Sort items by predicted scores in descending order
        scores_sorted_indices = np.argsort(-scores)
        print("scores_sorted_indices", scores_sorted_indices)

        # Select the top N recommended items
        self.inference_log.info("Select n-recommend items")
        return_score_list = scores_sorted_indices[:nrec_items]
        print("return_score_list", return_score_list)
        return return_score_list

    def run(self, user_id, movie_genre):
        self.load_predicting_data()

        if user_id is not None:
            rec_list = self.sample_recommendation_user(
                model=self.lightfm_model,
                interactions=self.lightfm_dataset["interactions"],
                user_id=user_id,
                user_dict=self.lightfm_dataset["user_id_map"],
                item_dict=self.lightfm_dataset["item_id_map"],
                threshold=4,
                nrec_items=10,
            )
            return rec_list
        
