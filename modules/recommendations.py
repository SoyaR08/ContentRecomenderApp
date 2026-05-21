from pyspark.sql import DataFrame, functions as F

def recommend_new_user(spark, movies, ratings_df, preferred_genres, top_n=20) -> DataFrame:
    """
    Recomendador para usuario nuevo (cold start) basado en géneros + popularidad global.

    Args:
        movies: DataFrame con movie_id, title, genres
        ratings_df: DataFrame de ratings original
        preferred_genres: lista de géneros elegidos por el usuario
        top_n: número de recomendaciones

    Returns:
        DataFrame con recomendaciones ordenadas
    """

    # ----------------------------
    # 1. Popularidad global de películas
    # ----------------------------
    movie_popularity = ratings_df.groupBy("movie_id") \
        .agg(F.avg("rating").alias("global_rating"))

    # ----------------------------
    # 2. Expandir géneros de películas
    # ----------------------------
    movie_genres = movies.select("movie_id", "title", "genres") \
        .withColumn("genre", F.explode(F.split("genres", ", "))) \
        .filter(F.col("genre") != "")

    # ----------------------------
    # 3. Score por coincidencia de género
    # ----------------------------
    preferred_genres_df = spark.createDataFrame(
        [(g,) for g in preferred_genres],
        ["genre"]
    )

    genre_match = movie_genres.join(
        preferred_genres_df,
        on="genre",
        how="inner"
    ).groupBy("movie_id") \
     .agg(F.count("*").alias("genre_match_score"))

    # ----------------------------
    # 4. Combinar todo
    # ----------------------------
    recs = movie_genres.select("movie_id", "title").distinct() \
        .join(genre_match, "movie_id", "left") \
        .join(movie_popularity, "movie_id", "left")

    # ----------------------------
    # 5. Score final (ajustable)
    # ----------------------------
    recs = recs.withColumn(
        "genre_match_score",
        F.coalesce(F.col("genre_match_score"), F.lit(0))
    ).withColumn(
        "global_rating",
        F.coalesce(F.col("global_rating"), F.lit(0))
    ).withColumn(
        "final_score",
        F.col("genre_match_score") * 2.0 + F.col("global_rating") * 0.5
    )

    # ----------------------------
    # 6. Top-N recomendaciones
    # ----------------------------
    return recs.orderBy(F.col("final_score").desc()).limit(top_n)

def ask_user_preferences(genre_cols):
    """
    Pregunta al usuario por consola sus géneros favoritos usando índices numéricos.
    Devuelve lista de géneros seleccionados.
    """

    print("\n🎬 Bienvenido al recomendador de películas")
    print("Selecciona tus géneros favoritos (separados por comas):\n")

    for i, genre in enumerate(genre_cols):
        print(f"{i}. {genre}")

    print("\nEjemplo: 0, 4, 7\n")

    while True:
        try:
            user_input = input("👉 Tus géneros: ")

            indices = [int(x.strip()) for x in user_input.split(",")]

            selected_genres = [
                genre_cols[i] for i in indices
                if 0 <= i < len(genre_cols)
            ]

            if not selected_genres:
                print("⚠️ No has seleccionado géneros válidos. Inténtalo otra vez.")
                continue

            return selected_genres

        except ValueError:
            print("⚠️ Entrada inválida. Usa números separados por comas.")