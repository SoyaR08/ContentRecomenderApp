from pyspark.sql import SparkSession
import os
from pyspark.sql import functions as F

from modules import recommend_new_user, ask_user_preferences
from preprocessing import build_movies_df, build_hybrid
from data import load_ratings_df

from models import build_als_model, build_content_scores

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

spark = SparkSession.builder.appName("debug").getOrCreate()

file_path = os.path.join(BASE_DIR, "ml-100k", "u.item")
ratings_path = os.path.join(BASE_DIR, "ml-100k", "u.data")

# ----------------------------
# MOVIES
# ----------------------------
movies = build_movies_df(spark, file_path)

genre_cols = [
    "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

# ----------------------------
# RATINGS
# ----------------------------
ratings_df = load_ratings_df(spark, ratings_path)

ratings_movies = ratings_df.join(
    movies.select("movie_id", "genres"),
    on="movie_id",
    how="inner"
)

# ----------------------------
# ALS MODEL
# ----------------------------

train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=42)

test_relevant = test_df.filter(F.col("rating") >= 4)

model = build_als_model(train_df)

als_recs = model.recommendForAllUsers(50)

als_flat = als_recs.selectExpr(
    "user_id",
    "explode(recommendations) as rec"
).select(
    "user_id",
    F.col("rec.movie_id"),
    F.col("rec.rating").alias("als_score")
)

K = 10

relevant_pairs = test_relevant.select(
    "user_id",
    "movie_id"
).withColumn("relevant", F.lit(1))

eval_df = als_flat.join(
    relevant_pairs,
    on=["user_id", "movie_id"],
    how="left"
).fillna({"relevant": 0})

precision_per_user = eval_df.groupBy("user_id").agg(
    (F.sum("relevant") / F.lit(K)).alias("precision_at_k")
)

# precision_at_k = precision_per_user.agg(
#     F.avg("precision_at_k").alias("precision@K")
# )

# precision_at_k.show()

# ----------------------------
# CONTENT MODEL
# ----------------------------

content_score = build_content_scores(ratings_movies, movies)

# ----------------------------
# HYBRID MODEL
# ----------------------------

hybrid, hybrid_ranked = build_hybrid(als_flat, content_score, k=10)

hybrid_eval = hybrid_ranked.join(
    relevant_pairs,
    on=["user_id", "movie_id"],
    how="left"
).fillna({"relevant": 0})

precision_hybrid = hybrid_eval.groupBy("user_id").agg(
    (F.sum("relevant") / F.lit(K)).alias("precision_at_k")
)

# precision_hybrid.agg(
#     F.avg("precision_at_k").alias("mean_precision_at_k")
# ).show()

# ----------------------------
# OUTPUT
# ----------------------------
final_recs = hybrid.join(
    movies.select("movie_id", "title"),
    on="movie_id",
    how="left"
)

final_recs.select(
    "user_id",
    "title",
    "als_score",
    "content_score",
    "final_score"
).orderBy(F.col("final_score").desc())#.show(20, truncate=False)

preferred_genres = ask_user_preferences(genre_cols)

recommend_new_user(
    spark,
    movies,
    ratings_df,
    preferred_genres,
    top_n=20
).show(truncate=False)