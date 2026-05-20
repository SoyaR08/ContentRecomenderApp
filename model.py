from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("recommender").getOrCreate()

ratings_df = spark.read.csv(
    "ml-100k/u.data",
    sep="\t",
    inferSchema=True
).toDF("user_id", "movie_id", "rating", "timestamp")

# ratings = spark.read.csv("ml-100k/u.data", sep="\t", inferSchema=True)

als = ALS(
    userCol="user_id",
    itemCol="movie_id",
    ratingCol="rating",
    coldStartStrategy="drop"
)

model = als.fit(ratings_df)

user_recs = model.recommendForAllUsers(10)
user_recs.show()