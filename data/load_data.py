from pyspark.sql import functions as DataFrame
from pyspark.sql import SparkSession


def load_movies_df(spark: SparkSession, file_path: str) -> DataFrame:
    movies = spark.read \
    .option("sep", "|") \
    .option("encoding", "ISO-8859-1") \
    .option("header", False) \
    .csv(file_path)

    return movies

def load_ratings_df(spark: SparkSession, ratings_path: str) -> DataFrame:

    ratings_df = spark.read.csv(
    ratings_path,
    sep="\t",
    inferSchema=True
    ).toDF("user_id", "movie_id", "rating", "timestamp")

    return ratings_df