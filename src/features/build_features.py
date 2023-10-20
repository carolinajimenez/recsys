"""
Recommendation System - Build Features
"""

import pandas as pd

from helpers.logger import Logger
from helpers.error_handler import CustomException


class FeatureCreation():
    """
    Feature Creation Class
    """

    def __init__(self) -> None:
        """
        Constructor method.
        """
        # LOG
        self.feature_log = Logger("FeatureCreation", show_asctime="dev").get_logger()

    def create_interaction_matrix(
            self,
            dataframe,
            user_col,
            item_col,
            rating_col,
            norm= False,
            threshold = None,
        ):
        """
        Create an interaction matrix dataframe from transactional type interactions

        Args:
            * dataframe (pd.Dataframe): User-item interactions
            * user_col (str): Column name containing user's identifier
            * item_col (str): Column name containing item's identifier
            * rating col (str): Column name containing user feedback on interaction with a given item
            * norm (bool): True if a normalization of ratings is needed
            * threshold (bool): Value above which the rating is favorable (Required if norm = True)

        Returns:
            * Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
        """
        interactions = dataframe.groupby([user_col, item_col])[rating_col] \
                .sum().unstack().reset_index(). \
                fillna(0).set_index(user_col)
        if norm:
            interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
        return interactions

    def merge_and_group_data(self, dataframe, id_column, column_names):
        """
        Merge and group data

        Args:
            * dataframe (pd.Dataframe): Pandas Dataframe for Users or Items
            * id_column (str): Name of table id
            * column_names (list): List of relevant columns

        Returns:
            pd.Dataframe: Group data
        """
        # Filter relevant columns (id_column and column_names)
        relevant_columns = [id_column] + column_names
        filtered_data = dataframe[relevant_columns]

        # Combine specified columns for each user/item using commas
        grouped_data = filtered_data.groupby(id_column, as_index=False).agg(lambda x: "-".join(map(str, x)))

        for col in column_names:
            # Delete duplicated values in the combined columns
            grouped_data[col] = grouped_data[col].apply(lambda x: "-".join(sorted(set(str(x).split("-")), key=str)))

        # Add a new column with values of entered columns separated by commas
        grouped_data["all_features"] = grouped_data[column_names].apply(lambda x: ", ".join(map(str, x)), axis=1)
        return grouped_data

    def create_features(self, dataframe, features_name, id_col_name):
        """
        Generate features that will be ready for feeding into lightfm

        Args:
            * dataframe (pd.Dataframe): Pandas Dataframe which contains features
            * features_name  (list): List of feature columns name avaiable in dataframe
            * id_col_name (str): Column name which contains id of the question or
                        answer that the features will map to.

        Returns:
            * (list): Containing process features that are ready for feed into LightFM
                        The format of each value will be
                        (user_id, ['feature_1', 'feature_2', 'feature_3'])
                        Ex. -> (1, ['military', 'army', '5'])
        """
        features = dataframe[features_name].apply(
            lambda x: ",".join(x.map(str)), axis=1)
        features = features.str.split(",")
        features = list(zip(dataframe[id_col_name], features))
        return features

    def generate_feature_list(self, dataframe, features_name):
        """
        Generate features list for mapping 

        Args:
            * dataframe (pd.Dataframe): Pandas Dataframe for Users or Q&A. 
            * features_name (list): List of feature columns name avaiable in dataframe. 
            
        Returns:
            * (pd.Series): List of all features for mapping 
        """
        features = dataframe[features_name].apply(
            lambda x: ",".join(x.map(str)), axis=1)
        features = features.str.split(",")
        features = features.apply(pd.Series).stack().reset_index(drop=True)
        return features

    def generate_features(self, logger, data_name, dataframe, id_column, column_names):
        """
        Generate features for users or items

        Args:
            * logger (logging.Logger): Logs
            * data_name (str): Name of dataframe: user or item
            * dataframe (pd.Dataframe): Pandas Dataframe for Users or Items
            * id_column (str): Name of table id
            * column_names (list): List of relevant columns

        Returns:
            * record_list (list): Record ids
            * record_features (list): Record-features
            * feature_list (pd.Series): List of all features for mapping 
        """
        if data_name == "user":
            # Adding the dummy user with ID 0
            dummy_user = {"userId": 0}
            for col in column_names:
                dummy_user[col] = None
        
            dataframe.loc[-1] = dummy_user
            dataframe.index = dataframe.index + 1
            dataframe = dataframe.sort_index()
            logger.info(" * Dummy user: %s", dummy_user)

        # Get list of records
        record_list = list(dataframe[id_column].unique())

        # First, combine and group the data
        df_grouped_data = self.merge_and_group_data(dataframe, id_column, column_names)
        # Then, create the relationship between record-features
        record_features = self.create_features(
            df_grouped_data, ["all_features"], id_column)
        # Get feature list
        feature_list = self.generate_feature_list(
            df_grouped_data, ["all_features"])

        return record_list, record_features, feature_list

    def get_formatted_interactions(
            self,
            df_interactions,
            user_id_col,
            item_id_col,
            interaction_col,
        ):
        """
        Get formatted interactions

        Args:
            * df_interactions (pd.DataFrame): Interactions between users and items.
            * user_id_col (str): The name of the column containing user identifiers.
            * item_id_col (str): The name of the column containing item identifiers.
            * interaction_col (str): The name of the column containing interaction information (e.g., ratings or scores).

        Returns:
            * interaction_tuples (list): A list of tuples representing formatted interactions.
                Each tuple contains three elements in the following order: (user_id, item_id, interaction_value).
                - user_id (int or str): The unique identifier of the user involved in the interaction.
                - item_id (int or str): The unique identifier of the item involved in the interaction.
                - interaction_value (float or int): The value of the interaction, typically a rating or score.
        """
        interaction_tuples = df_interactions.apply(
            lambda x: (x[user_id_col], x[item_id_col], x[interaction_col]), axis=1).tolist()
        return interaction_tuples

    def get_features(self, data):
        """
        Build features

        Args:    
            * data (dict):
                - user (dict): User info
                    - df_user (pd.Dataframe): User dataframe
                    - feature_columns (list): Column names
                - item (dict):
                    - df_item (pd.Dataframe): Item Dataframe
                    - feature_columns (list): Column names
                - interactions (pd.Dataframe): Interaction Dataframe

        Returns:
            * user_data (tuple):
                - user_list (list): List of user IDs
                - user_features_matrix (scipy.sparse._csr.csr_matrix): Features for each user
                - user_features_names (list): User features name
            * item_data (None | tuple):
                - item_list (list): List of item IDs
                - item_features_matrix (scipy.sparse._csr.csr_matrix): Features for each item
                - item_features_names (list): Item features name
            * formatted_interactions (list): Formatted interactions
        """
        self.feature_log.debug("Build user-item features and formatted interactions")
        try:
            # Get user and item features
            self.feature_log.debug(" -> Build user features")
            user_data = self.generate_features(
                logger=self.feature_log,
                data_name="user",
                dataframe=data["user"]["df_user"],
                id_column="userId",
                column_names=data["user"]["feature_columns"]
            )

            self.feature_log.debug(" -> Build item features")
            item_data = self.generate_features(
                logger=self.feature_log,
                data_name="item",
                dataframe=data["item"]["df_item"],
                id_column="movieId",
                column_names=data["item"]["feature_columns"]
            )

            df_interactions = data["interactions"]
            formatted_interactions = []
            if df_interactions is not None and not df_interactions.empty:
                formatted_interactions = self.get_formatted_interactions(
                    df_interactions=df_interactions,
                    user_id_col="userId",
                    item_id_col="movieId",
                    interaction_col="rating",
                )
            self.feature_log.debug("✔ User-item features and formatted interactions ready")
            return user_data, item_data, formatted_interactions
        except Exception as error:
            custom_exception = CustomException(
                f"✘ Error FeatureCreation-get features: {error}", self.feature_log)
            raise custom_exception from error
