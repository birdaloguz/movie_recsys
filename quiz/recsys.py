import pandas as pd
import numpy as np
from scipy import sparse
import scipy
from scipy.sparse.linalg import svds



def matrix_factorization(hist_user, offered_top, df_movies_org, df_ratings_org, U, sigma, Vt, movie_columns):

    u = np.array([np.array([0 for i in range(0, 13950)])])
    for i in hist_user:
        u[0][int(i)-1]=5.0

    u_prime = u.dot(Vt.T).dot(np.linalg.inv(sigma)).dot(sigma).dot(Vt).ravel()
    preds_df = pd.DataFrame(np.array([u_prime]), columns=movie_columns)
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

    return list(scores_top10.head(3)['title'])


def knn(hist_user, offered_top, df_movies_org, um_matrix, model_knn):

    #get distance for each movie in history and take the average for prediction
    avg_dict = {}
    for movie in hist_user:
        try:
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
    top_3 = [df_movies_org.loc[df_movies_org["movie_id"]==i].iloc[0]["title"] for i in top_3]
    return top_3

def neural_networks():
    #TODO
    pass