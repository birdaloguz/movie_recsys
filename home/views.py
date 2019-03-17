from django.views.generic import TemplateView
from django.shortcuts import render, redirect

from home.forms import HomeForm
from home.models import Post
from movies.models import Movie, Tag, Link, Rating
import pandas as pd

class HomeView(TemplateView):
    template_name = 'home/home.html'

    def get(self, request):
        form = HomeForm()
        posts = Post.objects.all().order_by('-date')

        df_ratings = pd.DataFrame(list(Rating.objects.all().values()))
        df_movies = pd.DataFrame(list(Movie.objects.all().values()))[['movie_id', 'title']]
        df_ratings = df_ratings[['movie_id', 'rating']].groupby(['movie_id']).size().reset_index(name='counts')
        df_ratings = df_ratings.sort_values(by=['counts'], ascending=False)
        df_ratings = df_ratings.head(1000)
        df_ratings = pd.merge(df_ratings, df_movies, how='left', on=['movie_id'])
        df = df_ratings.sample(n=10)
        random_movies = [{'movie_id': row['movie_id'], 'title': row['title']} for idx, row in df.iterrows()]
        args = {'form': form, 'posts': posts, 'movies': random_movies}
        return render(request, self.template_name, args)

    def post(self, request):
        if 'movie-form' in request.POST:
            checked_movies = request.POST.getlist('checks[]')
            print(checked_movies)
            return redirect('home:home')

        form = HomeForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.user = request.user
            post.save()

            text = form.cleaned_data['post']
            form = HomeForm()
            return redirect('home:home')

        args = {'form': form, 'text': text}
        return render(request, self.template_name, args)