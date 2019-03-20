from django.views import View
from django.shortcuts import render, redirect

from movies.models import Movie, Tag, Link, Rating
import pandas as pd

class quiz(View):

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
        if 'movie-form' in request.POST:
            checked_movies = request.POST.getlist('checks[]')
            args = {'movies': checked_movies}
            return render(request, 'quiz/quiz_history.html', args)

        return render(request)


