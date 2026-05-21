from pyspark.sql import functions as F
from pyspark.sql.window import Window

def build_hybrid(als_flat, content_score, k=10):

    hybrid = als_flat.join(
        content_score,
        ["user_id", "movie_id"],
        "left"
    )

    hybrid = hybrid.withColumn(
        "final_score",
        F.coalesce(F.col("als_score"), F.lit(0.0)) * 0.7 +
        F.coalesce(F.col("content_score"), F.lit(0.0)) * 0.3
    )

    hybrid = hybrid.groupBy("user_id", "movie_id").agg(
        F.max("als_score").alias("als_score"),
        F.max("content_score").alias("content_score"),
        F.max("final_score").alias("final_score")
    )

    w = Window.partitionBy("user_id").orderBy(F.col("final_score").desc())

    hybrid_ranked = hybrid.withColumn("rank", F.row_number().over(w)) \
        .filter(F.col("rank") <= k)

    return hybrid, hybrid_ranked