import pandas as pd
import numpy as np
from scipy import sparse
import scipy
import csv
from sklearn.metrics.pairwise import cosine_similarity

df_movies_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/movies.csv", skiprows=[0], names=["movie_id", "title", "genres"]).drop(columns=['genres'])
df_ratings_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/ratings.csv", skiprows=[0],  names=["user_id", "movie_id", "rating", "timestamp"]).drop(columns=['timestamp']).head(1000000)
df_link_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/links.csv", skiprows=[0],  names=["movie_id", "imdb_id", "tmdb_id"]).drop(columns=['tmdb_id'])



df_ratings = df_ratings_org[['movie_id', 'rating']].groupby(['movie_id']).size().reset_index(name='counts')
df_ratings = df_ratings.sort_values(by=['counts'], ascending=False)
popular = df_ratings
popular =popular.loc[popular["counts"]>=500]
df_ratings_org = df_ratings_org.loc[df_ratings_org["movie_id"].isin(popular["movie_id"].values)]

matrix_df = df_ratings_org.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
um_matrix = scipy.sparse.csr_matrix(matrix_df.values)


#model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
#model_knn.fit(um_matrix)

#TODO create triplets between popular movies
#Create triplets from pairwise matches
pairwise_distances = cosine_similarity(um_matrix)
movies = [i for i in list(matrix_df.index)]
for i in movies:
    print(df_movies_org.loc[df_movies_org["movie_id"]==i])
print(movies)
triplets = []
#print(list(matrix_df.index))
#print(len(pairwise_distances[0]))
#print(movies)


for idx, value in enumerate(pairwise_distances):
    print(idx)
    #get closest pairs to create triplets
    index = [list(matrix_df.index)[i] for i in np.argpartition(pairwise_distances[idx], -3)[-3:]]
    triplets.append(index)



with open("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/triplets.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(triplets)

