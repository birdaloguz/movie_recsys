from django.views import View
from django.shortcuts import render, redirect
import re, scipy, numpy as np, requests, pandas as pd
regex = re.compile(".*?\((.*?)\)")
from random import shuffle
from movies.models import Movie, Tag, Link, Rating
from quiz.recsys import matrix_factorization, knn
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

#load movies and ratings from dataset folder
df_movies_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/movies.csv", skiprows=[0], names=["movie_id", "title", "genres"]).drop(columns=['genres'])
df_ratings_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/ratings.csv", skiprows=[0],  names=["user_id", "movie_id", "rating", "timestamp"]).drop(columns=['timestamp']).head(1000000)
df_link_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/links.csv", skiprows=[0],  names=["movie_id", "imdb_id", "tmdb_id"]).drop(columns=['tmdb_id'])
triplets = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/triplets.csv", names=[1, 2, 3])

#df_movies_org = df_movies_org[df_movies_org["movie_id"].isin(df_ratings_org.movie_id.unique())]

#create movie-ratings matrix
matrix_df = df_ratings_org.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
um_matrix = scipy.sparse.csr_matrix(matrix_df.values)


#matrix factorization model
um_matrix_mf = scipy.sparse.csr_matrix(matrix_df.transpose().values)

movie_columns = matrix_df.transpose().columns
user_ratings_mean = np.mean(um_matrix_mf, axis=1)
R_demeaned = um_matrix_mf - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=1)
sigma = np.diag(sigma)


#knn model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
model_knn.fit(um_matrix)

#choose movies to offer from triplets
offered_movies = []
offered_hist_movies = []

movies_to_offer = triplets.sample(n=40)
#movies_to_offer = triplets.head(30)
for index, row in movies_to_offer.iterrows():
    triplet=[row[1], row[2], row[3]]
    shuffle(triplet)
    if triplet[0] not in offered_movies:
        offered_movies.append(triplet[0])
    if triplet[1] not in offered_hist_movies:
        offered_hist_movies.append(triplet[1])
    if triplet[2] not in offered_hist_movies:
        offered_hist_movies.append(triplet[2])

offered_movies = offered_movies[:20]
offered_hist_movies = offered_hist_movies[:60]

#get dataframes of movies will be offered
offered_movies = df_movies_org.loc[df_movies_org["movie_id"].isin(offered_movies)]
offered_movies = pd.merge(offered_movies, df_link_org, how='left', on=['movie_id'])

offered_hist_movies = df_movies_org.loc[df_movies_org["movie_id"].isin(offered_hist_movies)]
offered_hist_movies = pd.merge(offered_hist_movies, df_link_org, how='left', on=['movie_id'])

#TODO train model at start
class quiz_mf(View):
    #movies will be offered
    offered_top = []
    offered_hist = []
    #user selections
    top_user = []
    hist_user = []

    df_ratings = df_ratings_org
    df_movies = df_movies_org

    def get(self, request):
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title'], 'imdb_id': row['imdb_id']} for idx, row in offered_movies.iterrows()]

        #get poster links from OMDB API using movie titles
        random_movies=get_poster_links(random_movies)

        #return offers to user
        quiz_mf.offered_top = [movie["movie_id"] for movie in random_movies]
        args = {'movies': random_movies}
        return render(request, 'quiz/quiz_top10.html', args)

    def post(self, request):
        if 'movie-form1' in request.POST:
            #get user selections
            checked_top_movies = request.POST.getlist('checks[]')
            quiz_mf.top_user = checked_top_movies

            hist_movies = [{'movie_id': row['movie_id'], 'title': row['title'], 'imdb_id': row['imdb_id']} for idx, row in offered_hist_movies.iterrows()]

            # get poster links from OMDB API using movie titles
            hist_movies = get_poster_links(hist_movies)

            # return offers to user
            quiz_mf.offered_hist = [movie["movie_id"] for movie in hist_movies]
            args = {'movies': hist_movies}
            return render(request, 'quiz/quiz_history.html', args)

        if 'movie-form2' in request.POST:
            #get user selections
            checked_hist_movies = request.POST.getlist('checks[]')
            quiz_mf.hist_user = checked_hist_movies

            #compare results and return to user
            results = matrix_factorization(quiz_mf.hist_user, quiz_mf.offered_top, df_movies_org, df_ratings_org, U, sigma, Vt, movie_columns)
            top_user = [Movie.objects.filter(movie_id=i).values()[:1].get()['title'] for i in quiz_mf.top_user]
            args = {'results': results, "top_user": top_user}

            return render(request, 'quiz/quiz_results.html', args)

        return redirect('/quiz_mf')


class quiz_knn(View):
    # movies will be offered
    offered_top = []
    offered_hist = []
    # user selections
    top_user = []
    hist_user = []

    def get(self, request):
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title'], 'imdb_id': row['imdb_id']} for idx, row in
                         offered_movies.iterrows()]

        # get poster links from OMDB API using movie titles
        random_movies = get_poster_links(random_movies)

        # return offers to user
        quiz_knn.offered_top = [movie["movie_id"] for movie in random_movies]
        args = {'movies': random_movies}
        return render(request, 'quiz/quiz_top10.html', args)

    def post(self, request):
        if 'movie-form1' in request.POST:
            # get user selections
            checked_top_movies = request.POST.getlist('checks[]')
            quiz_knn.top_user = checked_top_movies

            hist_movies = [{'movie_id': row['movie_id'], 'title': row['title'], 'imdb_id': row['imdb_id']} for idx, row
                           in offered_hist_movies.iterrows()]

            # get poster links from OMDB API using movie titles
            hist_movies = get_poster_links(hist_movies)

            # return offers to user
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

#TODO try new methods
class quiz_nn(View):
    # movies will be offered
    offered_top = []
    offered_hist = []
    # user selections
    top_user = []
    hist_user = []

    def get(self, request):
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title'], 'imdb_id': row['imdb_id']} for idx, row in
                         offered_movies.iterrows()]

        # get poster links from OMDB API using movie titles
        random_movies = get_poster_links(random_movies)

        quiz_nn.offered_top = [movie["movie_id"] for movie in random_movies]
        args = {'movies': random_movies}
        return render(request, 'quiz/quiz_top10.html', args)

    def post(self, request):
        if 'movie-form1' in request.POST:
            checked_top_movies = request.POST.getlist('checks[]')
            quiz_nn.top_user = checked_top_movies

            hist_movies = [{'movie_id': row['movie_id'], 'title': row['title'], 'imdb_id': row['imdb_id']} for idx, row
                           in offered_hist_movies.iterrows()]

            # get poster links from OMDB API using movie titles
            hist_movies = get_poster_links(hist_movies)

            # return offers to user
            quiz_knn.offered_hist = [movie["movie_id"] for movie in hist_movies]
            args = {'movies': hist_movies}
            return render(request, 'quiz/quiz_history.html', args)

        if 'movie-form2' in request.POST:
            checked_hist_movies = request.POST.getlist('checks[]')
            quiz_nn.hist_user = checked_hist_movies

            results = knn(quiz_nn.hist_user, quiz_nn.offered_top, df_movies_org, um_matrix, model_knn)
            top_user=[Movie.objects.filter(movie_id=i).values()[:1].get()['title'] for i in quiz_nn.top_user]
            args = {'results': results, "top_user": top_user}

            return render(request, 'quiz/quiz_results.html', args)

        return redirect('/quiz_nn')

def get_poster_links(hist_movies):
    for idx, movie in enumerate(hist_movies):
        #movie_year = re.findall(regex, movie["title"])
        #movie_year = [i for i in movie_year if i.isdigit()][0]
        movie_name = re.sub("[\(\[].*?[\)\]]", "", movie["title"]).lower().replace(" ", "+")
        movie_name = movie_name.split(",")[0]
        movie_name = movie_name.replace(".", "")
        movie_name = movie_name.replace(":", "")
        req = "http://www.omdbapi.com/?t=" + movie_name + "&apikey=edc94b7d"
        r = requests.get(req)
        try:
            hist_movies[idx]["poster"] = r.json()["Poster"]
        except:
            hist_movies[idx]["poster"] = "NONE"
        hist_movies[idx]["title"] = re.sub("[\(\[].*?[\)\]]", "", movie["title"])
    return hist_movies