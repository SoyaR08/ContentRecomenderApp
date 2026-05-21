from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("recommender").getOrCreate()

ratings_df = spark.read.csv(
    "ml-100k/u.data",
    sep="\t",
    inferSchema=True
).toDF("user_id", "movie_id", "rating", "timestamp")

# ratings = spark.read.csv("ml-100k/u.data", sep="\t", inferSchema=True)

# als = ALS(
#     userCol="user_id",
#     itemCol="movie_id",
#     ratingCol="rating",
#     coldStartStrategy="drop"
# )

als = ALS(
    userCol="user_id",
    itemCol="movie_id",
    ratingCol="rating",
    coldStartStrategy="drop",
    rank=10,           # complejidad del modelo
    regParam=0.1,      # regularización
    maxIter=10
)

train, test = ratings_df.randomSplit([0.8, 0.2], seed=42)

# model = als.fit(ratings_df)
# user_recs = model.recommendForAllUsers(10)
# user_recs.show()

model = als.fit(train)

predictions = model.transform(test)
predictions.show()

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print("RMSE:", rmse)