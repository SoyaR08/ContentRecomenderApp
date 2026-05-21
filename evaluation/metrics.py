from pyspark.sql import functions as F
from pyspark.sql.window import Window

def precision_at_k(als_flat, test_relevant, K=10):

    # ----------------------------
    # 1. Top-K por usuario
    # ----------------------------
    w = Window.partitionBy("user_id").orderBy(F.col("als_score").desc())

    top_k = als_flat.withColumn(
        "rank",
        F.row_number().over(w)
    ).filter(F.col("rank") <= K)

    # ----------------------------
    # 2. Relevantes del test
    # ----------------------------
    relevant_pairs = test_relevant.select(
        "user_id",
        "movie_id"
    ).withColumn("relevant", F.lit(1))

    # ----------------------------
    # 3. Join con relevancia
    # ----------------------------
    eval_df = top_k.join(
        relevant_pairs,
        on=["user_id", "movie_id"],
        how="left"
    ).fillna({"relevant": 0})

    # ----------------------------
    # 4. Precision@K por usuario
    # ----------------------------
    precision_per_user = eval_df.groupBy("user_id").agg(
        (F.sum("relevant") / F.lit(K)).alias("precision_at_k")
    )

    # ----------------------------
    # 5. Media global
    # ----------------------------
    return precision_per_user.agg(
        F.avg("precision_at_k").alias("precision@K")
    )