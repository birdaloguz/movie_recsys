from django.views import View
from django.shortcuts import render, redirect

import re
regex = re.compile(".*?\((.*?)\)")

from movies.models import Movie, Tag, Link, Rating
import pandas as pd

from quiz.recsys import matrix_factorization, knn
from scipy import sparse
import scipy
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import requests

df_movies_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/movies.csv", skiprows=[0], names=["movie_id", "title", "genres"]).drop(columns=['genres'])
df_ratings_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/ratings.csv", skiprows=[0],  names=["user_id", "movie_id", "rating", "timestamp"]).drop(columns=['timestamp']).head(1000000)
df_link_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/links.csv", skiprows=[0],  names=["movie_id", "imdb_id", "tmdb_id"]).drop(columns=['tmdb_id'])
clustered_movies = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/clustered_movies.csv", skiprows=[0], names=["movie_id", "title", "cluster", "distances", "counts"])
clustered_movies = pd.merge(clustered_movies, df_link_org, how='left', on=['movie_id'])

df_movies_org = df_movies_org[df_movies_org["movie_id"].isin(df_ratings_org.movie_id.unique())]

matrix_df = df_ratings_org.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
um_matrix = scipy.sparse.csr_matrix(matrix_df.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
model_knn.fit(um_matrix)

offered_movies = df_movies_org.head(0)
offered_hist_movies = df_movies_org.head(0)
for i in range(0, 10):
    cluster_df=clustered_movies.loc[(clustered_movies["cluster"]==i)&(clustered_movies["counts"]>=100)].sort_values(by='distances')
    cluster_df = cluster_df.head(100)
    movie = cluster_df.sample(1)
    offered_movies = pd.concat([offered_movies, movie])
    cluster_df = cluster_df[cluster_df["movie_id"].isin(movie["movie_id"])==False]
    movie = cluster_df.sample(5)
    offered_hist_movies = pd.concat([offered_hist_movies, movie])

class quiz_mf(View):
    offered_top = []
    offered_hist = []
    top_user = []
    hist_user = []

    #df_movies_org = pd.DataFrame(list(Movie.objects.all().values()))[['movie_id', 'title', 'genre']]
    #df_ratings_org = pd.DataFrame(list(Rating.objects.all().values()))[['user_id', 'movie_id', 'rating']]

    df_ratings = df_ratings_org
    df_movies = df_movies_org

    df_link = df_link_org
    df_ratings = df_ratings[['movie_id', 'rating']].groupby(['movie_id']).size().reset_index(name='counts')
    df_ratings = df_ratings.sort_values(by=['counts'], ascending=False)
    df_ratings = df_ratings.head(1000)

    #popular_1000 = pd.merge(df_ratings, df_movies, how='left', on=['movie_id'])
    #popular_1000 = pd.merge(popular_1000, df_link, how='left', on=['movie_id'])
    #popular_1000 = pd.merge(df, df_link, how='left', on=['movie_id'])

    def get(self, request):
        #df = shuffle(quiz_mf.popular_1000).head(15)
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title'], 'imdb_id': row['imdb_id']} for idx, row in offered_movies.iterrows()]
        for movie in random_movies:
            movie_year=re.findall(regex, movie["title"])
            movie_year=[i for i in movie_year if i.isdigit()][0]
            movie_name=re.sub("[\(\[].*?[\)\]]", "", movie["title"]).lower().replace(" ", "+")
            req = "http://www.omdbapi.com/?t="+movie_name+"&apikey=edc94b7d"
            r = requests.get(req)
            print(r.json())


        quiz_mf.offered_top = [movie["movie_id"] for movie in random_movies]
        args = {'movies': random_movies}
        return render(request, 'quiz/quiz_top10.html', args)

    def post(self, request):
        if 'movie-form1' in request.POST:
            checked_top_movies = request.POST.getlist('checks[]')
            quiz_mf.top_user = checked_top_movies

            #df = shuffle(quiz_mf.popular_1000).head(70)

            hist_movies = [{'movie_id': row['movie_id'], 'title': row['title']} for idx, row in offered_hist_movies.iterrows()]
            quiz_mf.offered_hist = [movie["movie_id"] for movie in hist_movies]
            args = {'movies': hist_movies}
            return render(request, 'quiz/quiz_history.html', args)

        if 'movie-form2' in request.POST:
            checked_hist_movies = request.POST.getlist('checks[]')
            quiz_mf.hist_user = checked_hist_movies

            results = matrix_factorization(quiz_mf.hist_user, quiz_mf.offered_top, df_movies_org, df_ratings_org)
            top_user = [Movie.objects.filter(movie_id=i).values()[:1].get()['title'] for i in quiz_mf.top_user]
            args = {'results': results, "top_user": top_user}

            return render(request, 'quiz/quiz_results.html', args)

        return redirect('/quiz_mf')


class quiz_knn(View):
    offered_top = []
    offered_hist = []
    top_user = []
    hist_user = []

    #df_movies_org = pd.DataFrame(list(Movie.objects.all().values()))[['movie_id', 'title', 'genre']]
    #df_ratings_org = pd.DataFrame(list(Rating.objects.all().values()))[['user_id', 'movie_id', 'rating']].first(10000)

    df_ratings = df_ratings_org
    df_movies = df_movies_org

    df_link = df_link_org
    df_ratings = df_ratings[['movie_id', 'rating']].groupby(['movie_id']).size().reset_index(name='counts')
    df_ratings = df_ratings.sort_values(by=['counts'], ascending=False)
    df_ratings = df_ratings.head(1000)

    popular_1000 = pd.merge(df_ratings, df_movies, how='left', on=['movie_id'])
    #popular_1000 = pd.merge(df, df_link, how='left', on=['movie_id']).drop_duplicates()
    def get(self, request):
        #df = shuffle(quiz_knn.popular_1000).head(15)
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title']} for idx, row in offered_movies.iterrows()]
        quiz_knn.offered_top = [movie["movie_id"] for movie in random_movies]
        args = {'movies': random_movies}
        return render(request, 'quiz/quiz_top10.html', args)

    def post(self, request):
        if 'movie-form1' in request.POST:
            checked_top_movies = request.POST.getlist('checks[]')
            quiz_knn.top_user = checked_top_movies
            #df = shuffle(quiz_knn.popular_1000).head(70)
            hist_movies = [{'movie_id': row['movie_id'], 'title': row['title']} for idx, row in offered_hist_movies.iterrows()]
            quiz_knn.offered_hist = [movie["movie_id"] for movie in hist_movies]
            args = {'movies': hist_movies}
            return render(request, 'quiz/quiz_history.html', args)

        if 'movie-form2' in request.POST:
            checked_hist_movies = request.POST.getlist('checks[]')
            quiz_knn.hist_user = checked_hist_movies

            results = knn(quiz_knn.hist_user, quiz_knn.offered_top, df_movies_org, um_matrix, model_knn)
            top_user=[Movie.objects.filter(movie_id=i).values()[:1].get()['title'] for i in quiz_knn.top_user]
            args = {'results': results, "top_user": top_user}

            return render(request, 'quiz/quiz_results.html', args)

        return redirect('/quiz_knn')