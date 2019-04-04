from django.views import View
from django.shortcuts import render, redirect

from movies.models import Movie, Tag, Link, Rating
import pandas as pd

from quiz.recsys import matrix_factorization, knn

class quiz_mf(View):
    offered_top = []
    offered_hist = []
    top_user = []
    hist_user = []
    df_movies_org = pd.DataFrame(list(Movie.objects.all().values()))[['movie_id', 'title', 'genre']]
    df_ratings_org = pd.DataFrame(list(Rating.objects.all().values()))[['user_id', 'movie_id', 'rating']]

    df_ratings = df_ratings_org
    df_movies = df_movies_org

    df_link = pd.DataFrame(list(Link.objects.all().values()))[['imdb_id', 'movie_id']]
    df_ratings = df_ratings[['movie_id', 'rating']].groupby(['movie_id']).size().reset_index(name='counts')
    df_ratings = df_ratings.sort_values(by=['counts'], ascending=False)
    df_ratings = df_ratings.head(1000)

    df = pd.merge(df_ratings, df_movies, how='left', on=['movie_id'])
    popular_1000 = pd.merge(df, df_link, how='left', on=['movie_id'])

    def get(self, request):
        df = quiz_mf.popular_1000.sample(n=10)
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title']} for idx, row in df.iterrows()]
        quiz_mf.offered_top = [movie["movie_id"] for movie in random_movies]
        args = {'movies': random_movies}
        return render(request, 'quiz/quiz_top10.html', args)

    def post(self, request):
        if 'movie-form1' in request.POST:
            checked_top_movies = request.POST.getlist('checks[]')
            quiz_mf.top_user = checked_top_movies

            df = quiz_mf.popular_1000.sample(n=70)

            hist_movies = [{'movie_id': row['movie_id'], 'title': row['title']} for idx, row in df.iterrows()]
            quiz_mf.offered_hist = [movie["movie_id"] for movie in hist_movies]
            args = {'movies': hist_movies}
            return render(request, 'quiz/quiz_history.html', args)

        if 'movie-form2' in request.POST:
            checked_hist_movies = request.POST.getlist('checks[]')
            quiz_mf.hist_user = checked_hist_movies

            results = matrix_factorization(quiz_mf.hist_user, quiz_mf.offered_top, quiz_mf.df_movies_org, quiz_mf.df_ratings_org, quiz_mf.popular_1000)
            top_user = [Movie.objects.filter(movie_id=i).values()[:1].get()['title'] for i in quiz_mf.top_user]
            args = {'results': results, "top_user": top_user}

            return render(request, 'quiz/quiz_results.html', args)

        return redirect('/quiz_mf')


class quiz_knn(View):
    offered_top = []
    offered_hist = []
    top_user = []
    hist_user = []

    df_movies_org = pd.DataFrame(list(Movie.objects.all().values()))[['movie_id', 'title', 'genre']]
    df_ratings_org = pd.DataFrame(list(Rating.objects.all().values()))[['user_id', 'movie_id', 'rating']]

    df_ratings = df_ratings_org
    df_movies = df_movies_org

    df_link = pd.DataFrame(list(Link.objects.all().values()))[['imdb_id', 'movie_id']]
    df_ratings = df_ratings[['movie_id', 'rating']].groupby(['movie_id']).size().reset_index(name='counts')
    df_ratings = df_ratings.sort_values(by=['counts'], ascending=False)
    df_ratings = df_ratings.head(1000)
    df = pd.merge(df_ratings, df_movies, how='left', on=['movie_id'])
    popular_1000 = pd.merge(df, df_link, how='left', on=['movie_id'])
    def get(self, request):
        df = quiz_knn.popular_1000.sample(n=10)
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title']} for idx, row in df.iterrows()]
        quiz_knn.offered_top = [movie["movie_id"] for movie in random_movies]
        args = {'movies': random_movies}
        return render(request, 'quiz/quiz_top10.html', args)

    def post(self, request):
        if 'movie-form1' in request.POST:
            checked_top_movies = request.POST.getlist('checks[]')
            quiz_knn.top_user = checked_top_movies

            df = quiz_knn.popular_1000.sample(n=70)

            hist_movies = [{'movie_id': row['movie_id'], 'title': row['title']} for idx, row in df.iterrows()]
            quiz_knn.offered_hist = [movie["movie_id"] for movie in hist_movies]
            args = {'movies': hist_movies}
            return render(request, 'quiz/quiz_history.html', args)

        if 'movie-form2' in request.POST:
            checked_hist_movies = request.POST.getlist('checks[]')
            quiz_knn.hist_user = checked_hist_movies

            results = knn(quiz_knn.hist_user, quiz_knn.offered_top, quiz_knn.df_movies_org, quiz_knn.df_ratings_org, quiz_knn.popular_1000)
            top_user=[Movie.objects.filter(movie_id=i).values()[:1].get()['title'] for i in quiz_knn.top_user]
            args = {'results': results, "top_user": top_user}

            return render(request, 'quiz/quiz_results.html', args)

        return redirect('/quiz_knn')