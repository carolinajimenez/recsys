"""
Recommendation System POC
"""

import pandas as pd
from PIL import Image
import streamlit as st

from models.train_model import LightFMTrainer
from models.predict_model import LightFMPredictor


trainer = LightFMTrainer()
predictor = LightFMPredictor()


class RecommendationSystem():
    """
    Recommendation System
    """

    def __init__(self) -> None:
        """
        Constructor method.
        """
        # DATA
        trainer.get_data()
        self.users = trainer.users
        self.movies = trainer.movies

    def train_model(self):
        """
        Train LightFM model
        """
        res = trainer.run()
        return res

    def recommend(self, user_id=None, movie_genre=None):
        """
        LightFM Predictor

        Args:
            * user_id (str)
            * movie_genre (str)
        """
        rec_list = predictor.run(user_id, movie_genre, self.movies)
        return rec_list


if __name__ == "__main__":
    st.title("Recommendation System")
    recsys = RecommendationSystem()

    st.write("User data")
    st.write(recsys.users.head())
    st.write("Movie data")
    st.write(recsys.movies.head())

    model_image = Image.open("reports/figures/you_might_like.png")

    img, btn = st.columns([2.4, 1])
    img.image(model_image, caption='LightFM model')
    btn1_is_clicked = btn.button(label="Train model")

    if btn1_is_clicked:
        res = recsys.train_model()
        if not res:
            st.warning("There was an error in training", icon="⚠️")

    rec_list = None

    txt1, btn2 = st.columns([1.5, 1])
    txt1.text_input("User ID", key="user_id")
    btn2_is_clicked = btn2.button(label="Recommend by user id")
    if btn2_is_clicked:
        if not st.session_state.user_id:
            st.warning("User ID is empty", icon="⚠️")
        else:
            rec_list = recsys.recommend(user_id=st.session_state.user_id)

    txt2, btn3 = st.columns([1.5, 1])
    txt2.text_input("Movie genre", key="movie_genre")
    btn3_is_clicked = btn3.button(label="Recommend by movie genre")
    if btn3_is_clicked:
        if not st.session_state.movie_genre:
            st.warning("Movie genre is empty", icon="⚠️")
        else:
            rec_list = recsys.recommend(movie_genre=st.session_state.movie_genre)

    if rec_list is not None:
        print("rec_list", rec_list)
        st.write("Recommended Items:")
        # Create a DataFrame with the results
        results_df = pd.DataFrame({'movieId': rec_list})

        # Merge between the results DataFrame and the movies DataFrame
        recommended_movies = pd.merge(results_df, recsys.movies, on='movieId', how='inner')
        st.write(recommended_movies)
