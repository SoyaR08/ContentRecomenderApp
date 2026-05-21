from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from testingmodel import train_model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spark = SparkSession.builder.appName("debug").getOrCreate()

file_path = os.path.join(BASE_DIR, "ml-100k", "u.item")
ratings_path = os.path.join(BASE_DIR, "ml-100k", "u.data")

DATA_DIR = os.path.join(BASE_DIR, "ml-100k")

MOVIES_PATH = os.path.join(DATA_DIR, "u.item")
RATINGS_PATH = os.path.join(DATA_DIR, "u.data")
GENRES_PATH = os.path.join(DATA_DIR, "u.genre")

# ----------------------------
# Películas
# ----------------------------

movies = spark.read \
    .option("sep", "|") \
    .option("encoding", "ISO-8859-1") \
    .option("header", False) \
    .csv(file_path)

columns = [
    "movie_id", "title", "release_date", "video_release_date",
    "imdb_url", "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

movies = movies.toDF(*columns)
movies = movies.select(
    "movie_id", "title", "release_date", "unknown",
    "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
)

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
        *[
            F.when(F.col(c) == 1, F.lit(c))
            for c in genre_cols
        ]
    )
)

# ----------------------------
# RATINGS
# ----------------------------

ratings_df = spark.read.csv(
    ratings_path,
    sep="\t",
    inferSchema=True
).toDF("user_id", "movie_id", "rating", "timestamp")

ratings_movies = ratings_df.join(
    movies.select("movie_id", "title", "genres"),
    on="movie_id",
    how="inner"
)

# ----------------------------
# ALS MODEL
# ----------------------------

model = train_model(ratings_df)

als_recs = model.recommendForAllUsers(50)

als_flat = als_recs.selectExpr(
    "user_id",
    "explode(recommendations) as rec"
).select(
    "user_id",
    F.col("rec.movie_id"),
    F.coalesce(F.col("rec.rating"), F.lit(0.0)).alias("als_score")
)

# ----------------------------
# CONTENT MODEL
# ----------------------------

user_genres = ratings_movies \
    .select("user_id", "rating", F.explode(F.split("genres", ", ")).alias("genre"))

user_profile = user_genres.groupBy("user_id", "genre") \
    .agg(F.avg("rating").alias("avg_rating"))

# movie genres (IMPORTANTE: SOLO UNA EXPLOSIÓN)
movie_genres = movies.select("movie_id", "genres") \
    .withColumn("genre", F.explode(F.split("genres", ", "))) \
    .filter(
        (F.col("genre").isNotNull()) &
        (F.col("genre") != "")
    )

# número de géneros por película (SIN DUPLICADOS)
genre_counts = movie_genres.groupBy("movie_id") \
    .agg(F.count("genre").alias("genre_count"))

# raw score
raw_content_score = user_profile.join(
    movie_genres,
    on="genre"
).groupBy("user_id", "movie_id") \
 .agg(F.sum("avg_rating").alias("raw_content_score")) \
 .join(
    genre_counts,
    on="movie_id",
    how="left"
)

# SAFE CONTENT SCORE (sin nulls reales)
content_score = raw_content_score.withColumn(
    "content_score",
    F.least(
        F.coalesce(
            F.col("raw_content_score") / F.coalesce(F.col("genre_count"), F.lit(1)),
            F.lit(0.0)
        ),
        F.lit(5.0)
    )
)

# ----------------------------
# HYBRID MODEL
# ----------------------------


hybrid = als_flat.join(
    content_score,
    ["user_id", "movie_id"],
    "left"
)

hybrid = hybrid.withColumn(
    "final_score",
    (
        F.coalesce(F.col("als_score"), F.lit(0.0)) * 0.7 + F.coalesce(F.col("content_score"), F.lit(0.0)) * 0.3
    )
)

# Posible solución de chatgpt

hybrid = hybrid.groupBy(
    "user_id",
    "movie_id"
).agg(
    F.max("als_score").alias("als_score"),
    F.max("content_score").alias("content_score"),
    F.max("final_score").alias("final_score")
)

# ----------------------------
# OUTPUT
# ----------------------------

final_recs = hybrid.orderBy(F.col("final_score").desc())

# Esto pa mostrar el título de la peli

final_recs = final_recs.join(
    movies.select("movie_id", "title"),
    on="movie_id",
    how="left"
)

# final_recs.select(
#     "title",
#     "als_score",
#     "content_score",
#     "final_score"
# ).show(20, truncate=False)

final_recs.select(
    "user_id",
    "title",
    "als_score",
    "content_score",
    "final_score"
).orderBy(F.col("final_score").desc()) \
 .show(20, truncate=False)