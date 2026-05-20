import pandas as pd
from loader import load_data

data = load_data()


movie_matrix = data.pivot_table(
    index='user_id',
    columns='title',
    values='rating'
)

print(data.head())