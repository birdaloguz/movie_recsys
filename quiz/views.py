from django.views import View
from django.shortcuts import render, redirect

from movies.models import Movie, Tag, Link, Rating
import pandas as pd

class quiz(View):
    top_user = []
    hist_user = []

    def get(self, request):

        df_ratings = pd.DataFrame(list(Rating.objects.all().values()))
        df_movies = pd.DataFrame(list(Movie.objects.all().values()))[['movie_id', 'title']]
        df_ratings = df_ratings[['movie_id', 'rating']].groupby(['movie_id']).size().reset_index(name='counts')
        df_ratings = df_ratings.sort_values(by=['counts'], ascending=False)
        df_ratings = df_ratings.head(1000)
        df_ratings = pd.merge(df_ratings, df_movies, how='left', on=['movie_id'])
        df = df_ratings.sample(n=10)
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title']} for idx, row in df.iterrows()]
        args = {'movies': random_movies}
        return render(request, 'quiz/quiz_top10.html', args)

    def post(self, request):
        if 'movie-form1' in request.POST:
            checked_top_movies = request.POST.getlist('checks[]')
            quiz.top_user = checked_top_movies
            df_movies = pd.DataFrame(list(Movie.objects.all().values()))[['movie_id', 'title']]
            df = df_movies.sample(n=50)
            hist_movies = [{'movie_id': row['movie_id'], 'title': row['title']} for idx, row in df.iterrows()]
            args = {'movies': hist_movies}
            return render(request, 'quiz/quiz_history.html', args)

        if 'movie-form2' in request.POST:
            checked_hist_movies = request.POST.getlist('checks[]')
            quiz.hist_user = checked_hist_movies

            #TO DO: machine learning calculations here
            results = []
            top_user=[Movie.objects.filter(movie_id=i).values()[:1].get()['title'] for i in quiz.top_user]
            args = {'results': results, "top_user": top_user}
            return render(request, 'quiz/quiz_results.html', args)

        return redirect('/quiz/')


