import pandas as pd
import numpy as np
from scipy import sparse
import scipy
from scipy.sparse.linalg import svds



def matrix_factorization(hist_user, offered_top, df_movies_org, df_ratings_org, popular_1000):
    max_userID= df_ratings_org["user_id"].max()

    for movie in hist_user:
        df_ratings_org = df_ratings_org.append(pd.DataFrame([[max_userID+1, int(movie), 5.0]], columns=df_ratings_org.columns), ignore_index=True)

    matrix_df = df_ratings_org.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    # matrix_df = matrix_df[list(popular_1000["movie_id"])]
    um_matrix = scipy.sparse.csr_matrix(matrix_df.values)

    user_ratings_mean = np.mean(um_matrix, axis=1)
    R_demeaned = um_matrix - user_ratings_mean.reshape(-1, 1)

    U, sigma, Vt = svds(R_demeaned, k=1)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) #+ user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=matrix_df.columns)
    predictions = recommend_movies(preds_df, max_userID+1, df_movies_org, df_ratings_org, offered_top)

    return predictions


def recommend_movies(predictions_df, user_id, movies_df, original_ratings_df, offered_top):
    # Get and sort the user's predictions
    user_row_number = user_id - 1  # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)  # UserID starts at 1

    scores_top10 = movies_df[movies_df['movie_id'].isin(offered_top)]
    predictions_user = pd.DataFrame(sorted_user_predictions).reset_index().rename(index=str, columns={user_row_number: "prediction"})
    predictions_user['movie_id'] = predictions_user['movie_id'].astype(str).astype(int)
    scores_top10 = scores_top10.merge(predictions_user, how='left', left_on='movie_id', right_on='movie_id').sort_values(by='prediction', ascending=False)

    return list(scores_top10.head(3)['title'])


def knn(hist_user, offered_top, df_movies_org, um_matrix, model_knn):

    avg_dict = {}
    for movie in hist_user:
        try:
            distances, indices = model_knn.kneighbors(um_matrix[int(movie)-1], n_neighbors=13950)
            #sometimes index error
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

    for i, (idx, dist) in enumerate(raw_recommends):
        if idx in offered_top:
            predictions.append([idx, dist])
    top_3 = predictions[:3]
    top_3 = [i[0] for i in top_3]
    top_3 = [df_movies_org.loc[df_movies_org["movie_id"]==i].iloc[0]["title"] for i in top_3]
    return top_3