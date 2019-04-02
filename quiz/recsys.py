import pandas as pd
import numpy as np
from scipy import sparse
import scipy
from scipy.sparse.linalg import svds
from django.shortcuts import redirect
from movies.models import Movie, Tag, Link, Rating
from accounts.models import User

def test(hist_user, offered_top):
    df_movies = pd.DataFrame(list(Movie.objects.all().values()))[['movie_id', 'title', 'genre']]
    df_ratings = pd.DataFrame(list(Rating.objects.all().values()))[['user_id', 'movie_id', 'rating']]
    max_userID= df_ratings["user_id"].max()
    for movie in hist_user:
        df_ratings = df_ratings.append(pd.DataFrame([[max_userID+1, movie, 5.0]], columns=df_ratings.columns), ignore_index=True)

    matrix_df = df_ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    um_matrix = scipy.sparse.csr_matrix(matrix_df.values)

    user_ratings_mean= np.mean(um_matrix, axis=1)
    R_demeaned = um_matrix - user_ratings_mean.reshape(-1, 1)

    U, sigma, Vt = svds(R_demeaned, k = 50)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=matrix_df.columns)

    predictions = recommend_movies(preds_df, max_userID+1, df_movies, df_ratings, offered_top)

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