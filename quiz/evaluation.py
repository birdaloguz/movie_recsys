import re, scipy, numpy as np, requests, pandas as pd
from quiz.recsys import csr_matrix_indices
from scipy import sparse
from quiz.theano_bpr import BPR
from six.moves import cPickle
import os
from scipy.sparse.linalg import svds
from quiz.clustering import clustering
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle

from quiz.recsys import bpr, knn, matrix_factorization
from sklearn.model_selection import train_test_split

df_movies_org = pd.read_csv('/home/binglidev001/movie_recsys/dataset/movietweetings/movies.dat', sep='::', header=None, names=["movie_id", "title", "genre"])
df_ratings_org = pd.read_csv('/home/binglidev001/movie_recsys/dataset/movietweetings/ratings.dat', sep='::', header=None, names=["user_id", "movie_id", "rating", "timestamp"])




triplets = clustering(df_ratings_org)
triplets = pd.DataFrame(triplets, columns=[1, 2, 3])

df_movies_org = df_movies_org[df_movies_org["movie_id"].isin(df_ratings_org.movie_id.unique())].reset_index()

# create movie-ratings matrix
matrix_df = df_ratings_org.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
matrix_df_train, matrix_df_test = train_test_split(matrix_df, test_size=0.2)


um_matrix = scipy.sparse.csr_matrix(matrix_df_train.transpose().values)

# matrix factorization model
um_matrix_mf = scipy.sparse.csr_matrix(matrix_df_train.values)

movie_columns = matrix_df_train.columns
user_ratings_mean = np.mean(um_matrix_mf, axis=1)
R_demeaned = um_matrix_mf  # - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)

# knn model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
model_knn.fit(um_matrix)


#load existing bpr model
#f = open(os.path.dirname(os.path.abspath(__file__)) + '/models/bpr_model.save', 'rb')
#bpr_model = cPickle.load(f)
#f.close()




correlation_results=[]
test_users = []
for idx, row in matrix_df_test.iterrows():
    c = 0
    movie_user = []
    for index, i in enumerate(row):
        if i>5:
            movie_user.append(index)
            c=c+1
    if c>=8:
        test_users.append([idx, movie_user])
print(len(test_users))
print(test_users)

df_movies_org["index"] = df_movies_org.index

idx_dict = df_movies_org.set_index('movie_id').T.to_dict('list')
idx_reverse_dict = df_movies_org.set_index('index').T.to_dict('list')

for user in test_users:
    # choose movies to offer from triplets
    offered_movies = []
    offered_hist_movies = []
    movies_to_offer = triplets.sample(n=40)
    # movies_to_offer = triplets.head(30)
    for index, row in movies_to_offer.iterrows():
        triplet = [row[1], row[2], row[3]]
        triplet = shuffle(triplet)
        if triplet[0] not in offered_movies and triplet[0] not in offered_hist_movies:
            offered_movies.append(triplet[0])
        if triplet[1] not in offered_hist_movies and triplet[1] not in offered_movies:
            offered_hist_movies.append(triplet[1])
        if triplet[2] not in offered_hist_movies and triplet[2] not in offered_movies:
            offered_hist_movies.append(triplet[2])

    offered_movies = offered_movies[:17]
    for i in user[1][:3]:
        offered_movies.append(idx_reverse_dict[int(i)][0])
    users_picks = [idx_reverse_dict[int(i)][0] for i in user[1][:3]]
    hist_user = [str(idx_reverse_dict[int(i)][0]) for i in user[1][:3]]

    # get dataframes of movies will be offered
    offered_movies = df_movies_org.loc[df_movies_org["movie_id"].isin(offered_movies)]
    print("user")
    print(users_picks)
    offered_top = [row['movie_id'] for idx, row in offered_movies.iterrows()]
    #print(offered_top)
    results1 = matrix_factorization(hist_user, offered_top, df_movies_org, df_ratings_org, U, sigma, Vt, movie_columns)
    print(results1)
    results2 = knn(hist_user, offered_top, matrix_df.transpose(), um_matrix, model_knn, movie_columns, df_movies_org)
    print(results2)
    #results3 = bpr(hist_user, offered_top, df_movies_org, df_ratings_org, bpr_model, matrix_df.index)
    #print(results3)