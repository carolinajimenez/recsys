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
    btn_train_model_is_clicked = btn.button(label="Train model")

    txt1, btn2 = st.columns([1.5, 1])
    input_user_id_text = txt1.text_input("User ID", key="user_id")
    btn_user_id_is_clicked = btn2.button(label="Recommend by user id")

    txt2, btn3 = st.columns([1.5, 1])
    input_movie_genre_text = txt2.text_input("Movie genre", key="movie_genre")
    btn_movie_genre_is_clicked = btn3.button(label="Recommend by movie genre")

    if btn_train_model_is_clicked:
        res = recsys.train_model()
        if not res:
            st.warning("There was an error in training", icon="⚠️")

    rec_list = None

    if btn_user_id_is_clicked:
        input_movie_genre_text = ""
        if not st.session_state.user_id:
            st.warning("User ID is empty", icon="⚠️")
        else:
            rec_list = recsys.recommend(user_id=st.session_state.user_id)

    if btn_movie_genre_is_clicked:
        input_user_id_text = ""
        if not st.session_state.movie_genre:
            st.warning("Movie genre is empty", icon="⚠️")
        else:
            rec_list = recsys.recommend(movie_genre=st.session_state.movie_genre)

    if rec_list is not None:
        st.write("Recommended Items:")
        # Create a DataFrame with the results
        results_df = pd.DataFrame({'movieId': rec_list})

        # Merge between the results DataFrame and the movies DataFrame
        recommended_movies = pd.merge(results_df, recsys.movies, on='movieId', how='inner')
        st.write(recommended_movies)
