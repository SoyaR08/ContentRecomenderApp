import pandas as pd

def load_data():

    # Carga de ratings
    ratings_df = pd.read_csv(
        'ml-100k/u.data',
        sep='\t', # Separado por tabulaciones
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )

    # Carga de películas
    movies_df = pd.read_csv(
        'ml-100k/u.item',
        sep='|',
        encoding='latin-1', # Separado por tabulaciones
        header=None,
        usecols=[0, 1, 2],
        names=['movie_id', 'title', 'release_date']
    )

    data = pd.merge(ratings_df, movies_df, on='movie_id')

    return data