"""
Recommendation System - LigthFM Model Training
"""

import os
import random

import numpy as np
import pandas as pd
import pandas_profiling
from scipy import sparse
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k

from helpers.logger import Logger
from helpers.error_handler import CustomException
from helpers.save_files import save_file_pkl
from features.build_features import FeatureCreation


feature_creation = FeatureCreation()


class LightFMTrainer():
    """
    LigthFM Model Trainer Class
    """

    def __init__(self):
        """
        Constructor method.
        """
        # LOG
        self.training_log = Logger("LightFMTrainer", show_asctime="dev").get_logger()

        # DATA
        self.users = None
        self.movies = None
        self.ratings = None

        # MODEL INSTANCE
        self.model = LightFM(
            loss="warp",
            no_components=32,
            learning_rate=0.05,
            random_state=2019,
        )

        # REPORTS PATH
        self.reports_path = "recsys/reports"

    def load_data(self):
        """
        Load JSON data
        """
        data_path = "recsys/data/raw"
        self.movies = pd.read_csv(f"{data_path}/movies.csv")
        self.ratings = pd.read_csv(f"{data_path}/ratings.csv")

    def save_dataframe_locally(self, dataframe, db_path, db_name):
        """
        Save dataframe locally

        Args:
            * dataframe (pd.Dataframe): Dataframe to save
            * db_path (str): Dataset path
            * db_name (str): Dataset name
        """
        if dataframe is not None:
            os.makedirs(db_path, exist_ok=True)
            dataframe.to_csv(f"{db_path}/{db_name}.csv", index=False)

    def process_data(self):
        """
        Process data
        """
        #### #### ####
        # Data Inspection & Preparation: User data
        #### #### ####
        # Filter movies with rating greater than 4
        high_rated_movies = self.ratings[self.ratings["rating"] > 4]
        # Convert "rating" column to str before merge
        high_rated_movies["rating"] = high_rated_movies["rating"].astype(str)
        # Combine DataFrames to get movie genres with hight rating
        user_genre_preferences = high_rated_movies.merge(
            self.movies[["movieId", "genres"]], on="movieId", how="left")
        # Group and concatenate genres per user
        self.users = user_genre_preferences.groupby("userId")["genres"].apply(
            lambda x: "|".join(set(x.str.split("|").explode()))).reset_index()
        self.users.rename(columns={"genres": "genres_preference"}, inplace=True)

        self.save_dataframe_locally(
            dataframe=self.users,
            db_path="recsys/data/raw",
            db_name="users",
        )

    def validate_data(self, dataframe, df_name):
        """
        Validate dataframes

        Args:    
            * dataframe (pd.Dataframe): Dataframe
            * df_name (str): Dataframe name
        """
        self.training_log.debug(" * Validating dataframe: %s", df_name)
        # Verify dataframe are not null
        assert dataframe is not None, f"{df_name} is null"
        # Verify dataframe are not empty
        assert len(dataframe) > 0, f"{df_name} is empty"

    def get_data(self):
        """
        Get processed data
        """
        self.training_log.info("Get processed data")
        try:
            self.training_log.info(" -> Load data")
            self.load_data()
            self.training_log.info(" -> Process data")
            self.process_data()
            self.training_log.info(" -> Validate data")
            self.validate_data(self.users, "users")
            self.validate_data(self.movies, "movies")
            self.validate_data(self.ratings, "ratings")
            self.training_log.info("✔ Processed data ready")
        except Exception as error:
            custom_exception = CustomException(
                f"✘ Error LightFMTrainer-get data: {error}", self.training_log)
            raise custom_exception from error

    def validate_user_id_mapping(self, interactions, dataset):
        """
        Validate the user ID mapping

        Args:
            * interactions (list): Formatted interactions (list of tuples)
            * dataset (lightfm.data.Dataset): LightFM Dataset

        Returns:
            * dataset (lightfm.data.Dataset): Updated LightFM Dataset
        """
        self.training_log.debug(" -> Validate user ID mapping")
        # Obtain a list of all user_id in interactions
        # without repeated values
        user_ids_in_interactions = list(
            set(tupla[0] for tupla in interactions))

        user_ids = dataset.mapping()[0]
        # Search for missing user ids
        missing_users = [
            user_id for user_id in user_ids_in_interactions if user_id not in user_ids]
        if len(missing_users) == 0:
            self.training_log.debug(
                " * * All user IDs are present in the user ID mapping")
        else:
            self.training_log.debug(" * * Missing user IDs: %s", missing_users)
            dataset.fit_partial(users=missing_users)
            self.training_log.debug(
                " * * Adding missing user IDs to user ID mapping")

        return dataset

    def validate_item_id_mapping(self, interactions, dataset):
        """
        Validate the item ID mapping

        Args:
            * interactions (list): Formatted interactions (list of tuples)
            * dataset (lightfm.data.Dataset): LightFM Dataset

        Returns:
            * dataset (lightfm.data.Dataset): Updated LightFM Dataset
        """
        self.training_log.debug(" -> Validate item ID mapping")
        # Obtain a list of all item_id in interactions
        # without repeated values
        item_ids_in_interactions = list(
            set(tupla[1] for tupla in interactions))

        item_ids = dataset.mapping()[2]
        # Search for missing item ids
        missing_items = [
            item_id for item_id in item_ids_in_interactions if item_id not in item_ids]
        if len(missing_items) == 0:
            self.training_log.debug(
                " * * All item IDs are present in the item ID mapping")
        else:
            self.training_log.debug(" * * Missing item IDs: %s", missing_items)
            dataset.fit_partial(items=missing_items)
            self.training_log.debug(
                " * * Adding missing item IDs to item ID mapping")

        return dataset

    def build_lightfm_dataset(
        self,
        data,
        formatted_interactions,
    ):
        """
        Build LightFM dataset

        Args:
            * data (dict):
                - user_data (dict):
                    - user_list (list): List of user IDs
                    - user_features_matrix (scipy.sparse._csr.csr_matrix): Features for each user
                    - user_features_names (list): User features name
                - item_data (dict | None):
                    - df_item (pd.Dataframe): Item Dataframe
                    - item_list (list): List of item IDs
                    - item_features_matrix (scipy.sparse._csr.csr_matrix): Features for each item
                    - item_features_names (list): Item features name
            * formatted_interactions (list): Formatted interactions (list of tuples)

        Returns:
            * lightfm_dataset (dict): LightFM custom dataset
        """
        self.training_log.debug(" * Building LightFM dataset")
        # Get unique ids from item and user
        dataset = Dataset()
        dataset.fit(
            users=data["user_data"]["user_list"],
            items=data["item_data"]["item_list"],
            user_features=data["user_data"]["user_features_names"],
            item_features=data["item_data"]["item_features_names"],
        )

        # Validate if LightFM dataset contains all ids present in the
        # interactions
        dataset = self.validate_user_id_mapping(
            formatted_interactions, dataset)
        dataset = self.validate_item_id_mapping(
            formatted_interactions, dataset)

        # Building interactions
        interactions, weights = dataset.build_interactions(
            formatted_interactions)

        # Building matrix features
        user_features_matrix = dataset.build_user_features(
            data["user_data"]["user_features"])
        item_features_matrix = dataset.build_item_features(
            data["item_data"]["item_features"])

        # Building mappings: External (DB) to Internal (Model)
        user_id_map, _, item_id_map, _ = dataset.mapping()

        lightfm_dataset = {
            "interactions": interactions,
            "weights": weights,
            "user_features_matrix": user_features_matrix,
            "item_features_matrix": item_features_matrix,
            "user_feature_names": data["user_data"]["user_features_names"],
            "item_features_names": data["item_data"]["item_features_names"],
            "user_id_map": user_id_map,
            "item_id_map": item_id_map,
        }
        return lightfm_dataset

    def train_model(self, lightfm_dataset):
        """
        Train model

        Args:
            * lightfm_dataset (dict): LightFM custom dataset

        Returns:
            * model (lightfm.lightfm.LightFM): Trained LightFM model
        """
        self.training_log.debug(" -> Fit LightFM model")
        self.model.fit(
            interactions=lightfm_dataset["interactions"],
            sample_weight=lightfm_dataset["weights"],
            user_features=lightfm_dataset["user_features_matrix"],
            item_features=lightfm_dataset["item_features_matrix"],
            epochs=32,
            num_threads=4,
            verbose=True,
        )
        return self.model

    def evaluate_model(self, lightfm_dataset):
        """
        Evaluate model

        Args:
            * lightfm_dataset (dict): LightFM custom dataset

        Returns:
            * (dict):
                - {prefix}_score_auc (np.float32): Score AUC metric
                - {prefix}_prec_score (np.float32): Precision at k metric
        """
        self.training_log.debug(" * Evaluate trained LightFM model")
        score_auc = auc_score(
            self.model,
            lightfm_dataset["interactions"],
            user_features=lightfm_dataset["user_features_matrix"],
            item_features=lightfm_dataset["item_features_matrix"],
            num_threads=4,
        ).mean()

        prec_at_k = precision_at_k(
            self.model,
            lightfm_dataset["interactions"],
            user_features=lightfm_dataset["user_features_matrix"],
            item_features=lightfm_dataset["item_features_matrix"],
            k=5,
            num_threads=4,
        ).mean()

        self.training_log.debug(" * * Score (AUC): %s", score_auc)
        self.training_log.debug(" * * Precision (at k=5): %s", prec_at_k)

    def train_and_save_model(
        self,
        user_data,
        item_data,
        formatted_interactions,
    ):
        """
        Train LightFM model

        Args:
            * user_data (tuple):
                - user_list (list): List of user IDs
                - user_features_matrix (scipy.sparse._csr.csr_matrix): Features for each user
                - user_features_names (list): User features name
            * item_data (tuple):
                - df_item (pd.Dataframe): Item Dataframe
                - item_list (list): List of item IDs
                - item_features_matrix (scipy.sparse._csr.csr_matrix): Features for each item
                - item_features_names (list): Item features name
            * formatted_interactions (list): Formatted interactions

        Returns:
            * response (bool): Indicates if the process completed successfully
        """
        self.training_log.info("Training LightFM model")
        try:
            if item_data is None:
                self.training_log.warning("✘ There's not item data to train model")
                return False
            lightfm_dataset = self.build_lightfm_dataset(
                data={
                    "user_data": {
                        "user_list": user_data[0],
                        "user_features": user_data[1],
                        "user_features_names": user_data[2],
                    },
                    "item_data": {
                        "df_item": item_data[0],
                        "item_list": item_data[1],
                        "item_features": item_data[2],
                        "item_features_names": item_data[3],
                    },
                },
                formatted_interactions=formatted_interactions,
            )

            # Training and evaluation
            self.model = self.train_model(lightfm_dataset)
            self.evaluate_model(lightfm_dataset)

            save_file_pkl(
                file=lightfm_dataset,
                save_path="recsys/models",
                filename="lightfm_dataset",
            )
            save_file_pkl(
                file=self.model,
                save_path="recsys/models",
                filename="lightfm_model",
            )
            self.training_log.info("✔ Training model ready")
            return True
        except Exception as error:
            custom_exception = CustomException(
                f"✘ Error ModelTrainer-train_and_save_model: {error}", self.training_log)
            raise custom_exception from error

    def run(self):
        # Get user-item features and formatted interactions
        user_data, item_data, formatted_interactions = feature_creation.get_features(
            data={
                "user": {
                    "df_user": self.users,
                    "feature_columns": ["genres_preference"],
                },
                "item": {
                    "df_item": self.movies,
                    "feature_columns": ["genres"],
                },
                "interactions": self.ratings,
            }
        )

        item_data = (self.movies,) + item_data
        self.train_and_save_model(
            user_data=user_data,
            item_data=item_data,
            formatted_interactions=formatted_interactions,
        )
