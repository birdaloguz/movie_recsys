import pandas as pd
import numpy as np
from scipy import sparse
import scipy
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors

def matrix_factorization(hist_user, offered_top, df_movies_org, df_ratings_org, popular_1000):
    max_userID= df_ratings_org["user_id"].max()
    for movie in hist_user:
        df_ratings_org = df_ratings_org.append(pd.DataFrame([[max_userID+1, movie, 5.0]], columns=df_ratings_org.columns), ignore_index=True)

    matrix_df = df_ratings_org.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    matrix_df = matrix_df[list(popular_1000["movie_id"])]
    um_matrix = scipy.sparse.csr_matrix(matrix_df.values)
    print(matrix_df)

    user_ratings_mean= np.mean(um_matrix, axis=1)
    R_demeaned = um_matrix - user_ratings_mean.reshape(-1, 1)

    U, sigma, Vt = svds(R_demeaned, k = 50)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
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

def knn(hist_user, offered_top, df_movies_org, df_ratings_org, popular_1000):
    max_userID = df_ratings_org["user_id"].max()
    for movie in hist_user:
        df_ratings_org = df_ratings_org.append(
            pd.DataFrame([[max_userID + 1, movie, 5.0]], columns=df_ratings_org.columns), ignore_index=True)

    matrix_df = df_ratings_org.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    matrix_df = matrix_df[list(popular_1000["movie_id"])]
    matrix_df = matrix_df.transpose()
    um_matrix = scipy.sparse.csr_matrix(matrix_df.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
    model_knn.fit(um_matrix)

    distances, indices = model_knn.kneighbors([np.array(matrix_df.loc[max_userID])], n_neighbors=1000)
    predictions=[list(popular_1000["movie_id"])[i] for i in indices.tolist()[0]]
    predictions = [i for i in predictions if i in offered_top]
    top_3 = []
    for i in predictions[:3]:
        top_3.append(df_movies_org.loc[df_movies_org["movie_id"] == i]["title"].item())

    return top_3