from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, SparkSession


def train_model(ratings_df: DataFrame) -> ALSModel:
    als = ALS(
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        rank=10,
        regParam=0.1,
        maxIter=10,
        seed=42
    )
    model = als.fit(ratings_df)
    return model

def build_als_model(train_df) -> ALSModel:
    return train_model(train_df)