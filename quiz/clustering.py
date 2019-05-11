import pandas as pd
import numpy as np
from scipy import sparse
import scipy
import csv
from sklearn.metrics.pairwise import cosine_similarity


def clustering(df_ratings_org):

    df_ratings = df_ratings_org[['movie_id', 'rating']].groupby(['movie_id']).size().reset_index(name='counts')
    df_ratings = df_ratings.sort_values(by=['counts'], ascending=False)
    popular = df_ratings
    popular =popular.loc[popular["counts"]>=500]
    df_ratings_org = df_ratings_org.loc[df_ratings_org["movie_id"].isin(popular["movie_id"].values)]

    matrix_df = df_ratings_org.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
    um_matrix = scipy.sparse.csr_matrix(matrix_df.values)


    #Create triplets from pairwise matches
    pairwise_distances = cosine_similarity(um_matrix)
    triplets = []

    for idx, value in enumerate(pairwise_distances):
        #get closest pairs to create triplets
        index = [list(matrix_df.index)[i] for i in np.argpartition(pairwise_distances[idx], -3)[-3:]]
        triplets.append(index)

    return triplets

