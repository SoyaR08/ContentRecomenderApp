from pyspark.sql import functions as F

def build_content_scores(ratings_movies, movies):

    user_genres = ratings_movies \
        .select("user_id", "rating", F.explode(F.split("genres", ", ")).alias("genre")) \
        .filter(F.col("genre") != "")

    user_profile = user_genres.groupBy("user_id", "genre") \
        .agg(F.avg("rating").alias("avg_rating"))

    movie_genres = movies.select("movie_id", "genres") \
        .withColumn("genre", F.explode(F.split("genres", ", "))) \
        .filter(F.col("genre") != "")

    genre_counts = movie_genres.groupBy("movie_id") \
        .agg(F.countDistinct("genre").alias("genre_count"))

    raw = user_profile.join(movie_genres, on="genre") \
        .groupBy("user_id", "movie_id") \
        .agg(F.sum("avg_rating").alias("raw_content_score")) \
        .join(genre_counts, "movie_id", "left")

    content_score = raw.withColumn(
        "content_score",
        F.least(
            F.coalesce(
                F.col("raw_content_score") / F.coalesce(F.col("genre_count"), F.lit(1)),
                F.lit(0.0)
            ),
            F.lit(5.0)
        )
    )

    return content_score