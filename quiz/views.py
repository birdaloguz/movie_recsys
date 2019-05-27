from django.views import View
from django.shortcuts import render, redirect
import re, scipy, numpy as np, requests, pandas as pd
regex = re.compile(".*?\((.*?)\)")
from random import shuffle
from quiz.recsys import matrix_factorization, knn, bpr, csr_matrix_indices
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from quiz.clustering import clustering
from sklearn.utils import shuffle
from quiz.theano_bpr import BPR
from six.moves import cPickle
import os

def initializaton(movies_path, ratings_path, links_path, if_new_bpr):
    global df_movies_org, df_ratings_org, df_link_org, triplets, matrix_df, um_matrix, um_matrix_mf, U, sigma, Vt, movie_columns, model_knn, offered_movies, offered_hist_movies, bpr_model
    # load movies and ratings from dataset folder
    #df_movies_org = pd.read_csv(movies_path, skiprows=[0], names=["movie_id", "title", "genres"]).drop(columns=['genres'])
    #df_ratings_org = pd.read_csv(ratings_path, skiprows=[0], names=["user_id", "movie_id", "rating", "timestamp"]).drop(columns=['timestamp']).head(1000000)
    #df_ratings_org = pd.read_csv(ratings_path, skiprows=[0], names=["user_id", "movie_id", "rating", "timestamp"]).sort_values(by='timestamp').tail(1000000)
    df_link_org = pd.read_csv(links_path, skiprows=[0], names=["movie_id", "imdb_id", "tmdb_id"]).drop(columns=['tmdb_id'])

    df_movies_org = pd.read_csv('/home/binglidev001/movie_recsys/dataset/movietweetings/movies.dat', sep='::', header=None,
                                names=["movie_id", "title", "genre"])
    df_ratings_org = pd.read_csv('/home/binglidev001/movie_recsys/dataset/movietweetings/ratings.dat', sep='::', header=None,
                                 names=["user_id", "movie_id", "rating", "timestamp"])

    triplets = clustering(df_ratings_org)
    triplets = pd.DataFrame(triplets, columns=[1, 2, 3])

    df_movies_org = df_movies_org[df_movies_org["movie_id"].isin(df_ratings_org.movie_id.unique())].reset_index()

    # create movie-ratings matrix
    matrix_df = df_ratings_org.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
    um_matrix = scipy.sparse.csr_matrix(matrix_df.values)

    # matrix factorization model
    um_matrix_mf = scipy.sparse.csr_matrix(matrix_df.transpose().values)

    movie_columns = matrix_df.transpose().columns
    user_ratings_mean = np.mean(um_matrix_mf, axis=1)
    R_demeaned = um_matrix_mf - user_ratings_mean.reshape(-1, 1)

    U, sigma, Vt = svds(R_demeaned, k=1)
    sigma = np.diag(sigma)

    # knn model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
    model_knn.fit(um_matrix)

    # bpr model
    train_data = []
    for x in csr_matrix_indices(um_matrix_mf):
        train_data.append(x)

    if if_new_bpr:
        # Initialising BPR model, 10 latent factors
        bpr_model = BPR(5, len(matrix_df.transpose().index), len(matrix_df.index))
        # Training model, 30 epochs
        bpr_model.train(train_data, epochs=1)

        #save bpr model
        f = open(os.path.dirname(os.path.abspath(__file__)) + '/models/bpr_model.save', 'wb')
        cPickle.dump(bpr_model, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
    else:
        #load existing bpr model
        f = open(os.path.dirname(os.path.abspath(__file__)) + '/models/bpr_model.save', 'rb')
        bpr_model = cPickle.load(f)
        f.close()

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

    offered_movies = offered_movies[:20]
    offered_hist_movies = offered_hist_movies[:60]

    # get dataframes of movies will be offered
    offered_movies = df_movies_org.loc[df_movies_org["movie_id"].isin(offered_movies)]
    offered_movies = pd.merge(offered_movies, df_link_org, how='left', on=['movie_id'])

    offered_hist_movies = df_movies_org.loc[df_movies_org["movie_id"].isin(offered_hist_movies)]
    offered_hist_movies = pd.merge(offered_hist_movies, df_link_org, how='left', on=['movie_id'])

dataset_folder = os.path.dirname(os.path.dirname(__file__))+"/dataset/ml-20m/ml-20m/"

initializaton(dataset_folder+"movies.csv", dataset_folder+"ratings.csv", dataset_folder+"links.csv", False)


class quiz_mf(View):
    #movies will be offered
    offered_top = []
    offered_top_dict = []
    offered_hist = []
    #user selections
    top_user = []
    hist_user = []

    global df_movies_org, df_ratings_org, df_link_org, triplets, matrix_df, um_matrix, um_matrix_mf, U, sigma, Vt, movie_columns, model_knn, offered_movies, offered_hist_movies

    def get(self, request):
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title'], 'imdb_id': row['imdb_id']} for idx, row in offered_movies.iterrows()]

        #get poster links from OMDB API using movie titles
        random_movies=get_poster_links(random_movies)
        quiz_mf.offered_top_dict=random_movies
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
            results = [[item["title"], item["poster"]] for item in quiz_mf.offered_top_dict if item["movie_id"] in results]
            top_user =[[item["title"], item["poster"]] for item in quiz_mf.offered_top_dict if str(item["movie_id"]) in quiz_mf.top_user]
            args = {'results': results, "top_user": top_user}

            return render(request, 'quiz/quiz_results.html', args)

        return redirect('/quiz_mf')


class quiz_knn(View):
    # movies will be offered
    offered_top = []
    offered_top_dict = []
    offered_hist = []
    # user selections
    top_user = []
    hist_user = []

    global df_movies_org, df_ratings_org, df_link_org, triplets, matrix_df, um_matrix, um_matrix_mf, U, sigma, Vt, movie_columns, model_knn, offered_movies, offered_hist_movies

    def get(self, request):
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title'], 'imdb_id': row['imdb_id']} for idx, row in
                         offered_movies.iterrows()]

        # get poster links from OMDB API using movie titles
        random_movies = get_poster_links(random_movies)
        quiz_knn.offered_top_dict = random_movies
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

            results = knn(quiz_knn.hist_user, quiz_knn.offered_top, matrix_df, um_matrix, model_knn, movie_columns, df_movies_org)
            results = [[item["title"], item["poster"]] for item in quiz_knn.offered_top_dict if item["movie_id"] in results]
            top_user =[[item["title"], item["poster"]] for item in quiz_knn.offered_top_dict if str(item["movie_id"]) in quiz_knn.top_user]
            args = {'results': results, "top_user": top_user}

            return render(request, 'quiz/quiz_results.html', args)

        return redirect('/quiz_knn')


class quiz_bpr(View):
    # movies will be offered
    offered_top = []
    offered_top_dict = []
    offered_hist = []
    # user selections
    top_user = []
    hist_user = []

    global df_movies_org, df_ratings_org, df_link_org, triplets, matrix_df, um_matrix, um_matrix_mf, U, sigma, Vt, movie_columns, model_knn, offered_movies, offered_hist_movies, bpr_model

    def get(self, request):
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title'], 'imdb_id': row['imdb_id']} for idx, row in
                         offered_movies.iterrows()]

        # get poster links from OMDB API using movie titles
        random_movies = get_poster_links(random_movies)
        quiz_bpr.offered_top_dict = random_movies

        # return offers to user
        quiz_bpr.offered_top = [movie["movie_id"] for movie in random_movies]
        args = {'movies': random_movies}
        return render(request, 'quiz/quiz_top10.html', args)

    def post(self, request):
        if 'movie-form1' in request.POST:
            # get user selections
            checked_top_movies = request.POST.getlist('checks[]')
            quiz_bpr.top_user = checked_top_movies

            hist_movies = [{'movie_id': row['movie_id'], 'title': row['title'], 'imdb_id': row['imdb_id']} for idx, row
                           in offered_hist_movies.iterrows()]

            # get poster links from OMDB API using movie titles
            hist_movies = get_poster_links(hist_movies)

            # return offers to user
            quiz_bpr.offered_hist = [movie["movie_id"] for movie in hist_movies]
            args = {'movies': hist_movies}
            return render(request, 'quiz/quiz_history.html', args)

        if 'movie-form2' in request.POST:
            checked_hist_movies = request.POST.getlist('checks[]')
            quiz_bpr.hist_user = checked_hist_movies

            results = bpr(quiz_bpr.hist_user, quiz_bpr.offered_top, df_movies_org, df_ratings_org, bpr_model, matrix_df.index)
            results = [[item["title"], item["poster"]] for item in quiz_bpr.offered_top_dict if item["movie_id"] in results]
            top_user =[[item["title"], item["poster"]] for item in quiz_bpr.offered_top_dict if str(item["movie_id"]) in quiz_bpr.top_user]
            args = {'results': results, "top_user": top_user}

            return render(request, 'quiz/quiz_results.html', args)

        return redirect('/quiz_bpr')

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

class load_dataset(View):
    def get(self, request):

        return render(request, "quiz/load_dataset.html")

    def post(self, request):

        dataset_folder = request.POST.get('dataset')

        initializaton(dataset_folder + "/movies.csv", dataset_folder + "/ratings.csv", dataset_folder + "/links.csv", True)
        return render(request, "quiz/load_dataset.html")