recsys - Recommendation System POC
==============================

This project is an interactive movie recommendation application developed using Python, Streamlit, and the LightFM recommendation library. The app provides users with personalized movie recommendations based on their preferences and historical interactions.

Project Organization
------------

      recsys
      ├── LICENSE
      ├── README.md                       <- The top-level README for developers using this project.
      ├── data
      │   ├── processed                   <- The final, canonical data sets for modeling.
      │   │   └── users.csv
      │   └── raw                         <- The original, immutable data dump.
      │       ├── README.txt
      │       ├── links.csv
      │       ├── movies.csv
      │       ├── ratings.csv
      │       └── tags.csv
      ├── models                          <- Trained and serialized model and dataset.
      │   ├── lightfm_dataset
      │   └── lightfm_model
      ├── reports                         <- Generated graphics and figures to be used in reporting.
      │   └── figures
      │       └── you_might_like.png
      ├── requirements.txt                <- The requirements file for reproducing the analysis environment.
      └── src                             <- Source code for use in this project.
         ├── __init__.py
         ├── app.py                       <- Integrates model training, inference, and creates a Streamlit app for testing the recommendation system.
         ├── data                         <- Scripts to download or generate data.
         │   ├── __init__.py
         │   └── make_dataset.py
         ├── features                     <- Scripts to turn raw data into features for modeling.
         │   ├── __init__.py
         │   └── build_features.py
         ├── helpers                      <- Provide functions for handling errors, logging events, and managing files.
         │   ├── error_handler.py
         │   ├── logger.py
         │   └── save_files.py
         └── models                       <- Scripts to train models and then use trained models to make predictions.
            ├── __init__.py
            ├── predict_model.py
            └── train_model.py

Usage
------------

1. Clone the repository:
```
git clone https://github.com/carolinajimenez/recsys
cd recsys
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download movie dataset
```
python3 src/data/make_dataset.py
```

4. Run the Streamlit app:
```
streamlit run src/app.py
```


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
