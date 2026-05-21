from pyspark.sql import functions as F
from data import load_movies_df

def build_movies_df(spark, file_path):
    """
    Carga y transforma el dataset de películas (MovieLens 100K).
    Devuelve DataFrame con columnas limpias + columna genres.
    """

    movies = load_movies_df(spark, file_path)

    columns = [
        "movie_id", "title", "release_date", "video_release_date",
        "imdb_url", "unknown", "Action", "Adventure", "Animation",
        "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]

    movies = movies.toDF(*columns)

    genre_cols = [
        "unknown", "Action", "Adventure", "Animation",
        "Children", "Comedy", "Crime", "Documentary",
        "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]

    movies = movies.withColumn(
        "genres",
        F.concat_ws(
            ", ",
            *[F.when(F.col(c) == 1, F.lit(c)) for c in genre_cols]
        )
    )

    return movies