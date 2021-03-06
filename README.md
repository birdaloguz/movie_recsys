# Movie Recommendation Systems

This application is a Django based movie recommendation system. It includes various collaborative filtering methods and flexible for any user/rating dataset. For this demo [Movie Tweetings](https://github.com/sidooms/MovieTweetings) is used. It contains over 60,000 ratings provided by more than 12,000 users on 8,000 unique items. 

Here are the collaborative filtering methods used:
* KNN
* Matrix Factorization
* [Bayesian Personalised Ranking](https://arxiv.org/pdf/1205.2618.pdf) 

###### _&copy;2014 British Broadcasting Corporation and contributors_

## Flexible Dataset
Dataset loading page is available in the application. Any user item rating dataset can be used to use the application. 
But the format should be like MovieLens dataset. MovieLens dataset would be loaded if you do not load any dataset after running the server.
In dataset loading page there is an input for the dataset folder path. This path should include "movies.csv", "ratings.csv" and "links.csv".
Columns of the csv files should be:

* movies.csv: movie_id, title
* ratings.csv: user_id, movie_id, rating
* links.csv: movie_id, imdb_id

## Requirements

* [Python 3.6](https://www.python.org/downloads/release/python-360/)
* [Django Web Framework 1.11.20](https://docs.djangoproject.com/en/2.2/releases/1.11.20/)

## How to run the application

Setup packages in requirements.txt

```bash
$ cd movie_recsys/
$ python manage.py makemigrations
$ python manage.py migrate
$ python manage.py createsuperuser
$ python manage.py runserver
```


