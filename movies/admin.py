from django.contrib import admin

from movies.models import Tag, Movie, Link, Rating

admin.site.register(Tag)
admin.site.register(Movie)
admin.site.register(Link)
admin.site.register(Rating)