import re, scipy, numpy as np, requests, pandas as pd
from quiz.recsys import csr_matrix_indices
from scipy import sparse
from quiz.theano_bpr import BPR
from six.moves import cPickle
import os

#load movies and ratings from dataset folder
df_movies_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/movies.csv", skiprows=[0], names=["movie_id", "title", "genres"]).drop(columns=['genres'])
df_ratings_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/ratings.csv", skiprows=[0],  names=["user_id", "movie_id", "rating", "timestamp"]).sort_values(by='timestamo').tail(1000000)

#create movie-ratings matrix
matrix_df = df_ratings_org.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
um_matrix = scipy.sparse.csr_matrix(matrix_df.values)


um_matrix_mf = scipy.sparse.csr_matrix(matrix_df.transpose().values)
train_data = []
for x in csr_matrix_indices(um_matrix_mf):
    train_data.append(x)

# Initialising BPR model, 10 latent factors
bpr = BPR(10, len(matrix_df.transpose().index), len(matrix_df.index))
# Training model, 30 epochs
bpr.train(train_data, epochs=30)


f = open(os.path.dirname(os.path.abspath(__file__)) + '/models/bpr_model.save', 'wb')
cPickle.dump(bpr, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
