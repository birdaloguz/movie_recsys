import pandas as pd
import numpy as np
from scipy import sparse
import scipy
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from quiz.theano_bpr import BPR

def matrix_factorization(hist_user, offered_top, df_movies_org, df_ratings_org, U, sigma, Vt, movie_columns):
    #create new user vector from selected history set

    df_movies_org = df_movies_org.reset_index()

    u = np.array([np.array([0 for i in range(0, len(movie_columns))])])
    for i in hist_user:
        u[0][df_movies_org.loc[df_movies_org["movie_id"] == int(i)].index.astype(int)[0]]=5.0

    #calculate predictions for new user
    u_prime = u.dot(Vt.T).dot(np.linalg.inv(sigma)).dot(sigma).dot(Vt).ravel()
    preds_df = pd.DataFrame(np.array([u_prime]), columns=movie_columns)

    #get top 3 movie in offered set with the highest prediction
    predictions = recommend_movies(preds_df, df_movies_org, df_ratings_org, offered_top)

    return predictions


def recommend_movies(predictions_df, movies_df, original_ratings_df, offered_top):
    # Get and sort the user's predictions
    sorted_user_predictions = predictions_df.iloc[0].sort_values(ascending=False)  # UserID starts at 1

    # get the scores of offered movies and return top3 for prediction
    scores_top10 = movies_df[movies_df['movie_id'].isin(offered_top)]
    predictions_user = pd.DataFrame(sorted_user_predictions).reset_index().rename(index=str, columns={0: "prediction"})
    predictions_user['movie_id'] = predictions_user['movie_id'].astype(str).astype(int)
    scores_top10 = scores_top10.merge(predictions_user, how='left', left_on='movie_id', right_on='movie_id').sort_values(by='prediction', ascending=False)

    return list(scores_top10.head(3)['movie_id'])


def knn(hist_user, offered_top, df_movies_org, um_matrix, model_knn):

    #get distance for each movie in history and take the average for prediction
    avg_dict = {}
    for movie in hist_user:
        try:
            #get item based predictions for each item in history and sum the distances
            distances, indices = model_knn.kneighbors(um_matrix[int(movie)-1], n_neighbors=13950)
            distances = distances.squeeze().tolist()
            indices = indices.squeeze().tolist()
            for i in range(0, len(indices)):
                if str(indices[i] not in avg_dict):
                    avg_dict[str(indices[i])] = distances[i]
                else:
                    avg_dict[str(indices[i])] += distances[i]
        except:
            print(str(movie) + " index out of range!!!")
    indices = []
    distances = []
    #dict to list
    for key, value in avg_dict.items():
        indices.append(int(key))
        distances.append(value)

    raw_recommends = sorted(list(zip(indices, distances)), key=lambda x: x[1])[:0:-1]
    predictions = []

    #get the distances of offered movies and return top3 for prediction
    for i, (idx, dist) in enumerate(raw_recommends):
        if idx in offered_top:
            predictions.append([idx, dist])
    top_3 = predictions[:3]
    top_3 = [i[0] for i in top_3]
    return top_3

def bpr(hist_user, offered_top, df_movies_org, df_ratings_org, bpr_model, movie_indices):
    matrix_df = df_ratings_org.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
    new_user_id = df_ratings_org['user_id'].max()+1
    for movie in hist_user:
        df_ratings_org = df_ratings_org.append({"user_id": new_user_id, "movie_id": movie, "rating": 5.0}, ignore_index=True)
    um_matrix = scipy.sparse.csr_matrix(matrix_df.values)
    pairwise_distances = cosine_similarity(um_matrix)
    similar_user_id = heapq.nlargest(1, heapq.nlargest(1, range(len(pairwise_distances[-1][:-1])), key=pairwise_distances[-1][:-1].__getitem__))[0]+1
    bpr_predictions = bpr_model.predictions(similar_user_id)
    indices = list(movie_indices)

    raw_recommends = sorted(list(zip(indices, bpr_predictions)), key=lambda x: x[1])[:0:-1]
    predictions = []

    # get the distances of offered movies and return top3 for prediction
    for i, (idx, dist) in enumerate(raw_recommends):
        if idx in offered_top:
            predictions.append([idx, dist])
    top_3 = predictions[:3]
    top_3 = [i[0] for i in top_3]
    return top_3


def csr_matrix_indices(S):
    """
    Return a list of the indices of nonzero entries of a csr_matrix S
    """
    major_dim, minor_dim = S.shape
    minor_indices = S.indices

    major_indices = np.empty(len(minor_indices), dtype=S.indices.dtype)
    scipy.sparse._sparsetools.expandptr(major_dim, S.indptr, major_indices)

    return zip(major_indices, minor_indices)