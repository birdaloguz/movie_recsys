from django.db import models
from django.contrib.postgres.fields import JSONField

class Movie(models.Model):
    movie_id = models.PositiveIntegerField(primary_key=True)
    title = models.CharField(max_length=100)
    genre = JSONField()

class Link(models.Model):
    movie_id = models.PositiveIntegerField()
    imdb_id = models.PositiveIntegerField()
    tmdb_id = models.PositiveIntegerField()

class Rating(models.Model):
    user_id = models.PositiveIntegerField()
    movie_id = models.PositiveIntegerField()
    rating = models.FloatField()
    timestamp = models.DateTimeField()

class Tag(models.Model):
    user_id = models.PositiveIntegerField()
    movie_id = models.PositiveIntegerField()
    tag = models.CharField(max_length=40)
    timestamp = models.DateTimeField()



