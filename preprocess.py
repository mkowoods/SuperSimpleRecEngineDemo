import os
import pandas as pd
import json
import requests
import zipfile
import StringIO
import config

MOVIE_LENS_DIR = os.path.join(config.BASE_DIR, 'ml-latest-small')
DATA_DIR = os.path.join(config.BASE_DIR, 'data')
print DATA_DIR


def get_movies_data():
    print 'loading movies'
    movies = pd.read_csv(os.path.join(MOVIE_LENS_DIR, 'movies.csv'))
    movies_data = movies.T.to_dict().values()

    movie_dict = {}

    for movie in movies_data:
        movie['genres'] = movie['genres'].split('|')
        movie_dict[movie['movieId']] = movie

    return movie_dict


def get_ratings_data():
    print 'ratings data'
    ratings = pd.read_csv(os.path.join(MOVIE_LENS_DIR, 'ratings.csv'))
    ratings_data = ratings.T.to_dict().values()  # list of dicts [ {movieId, rating, timestamp, userId }

    rating_dict = {}

    # ids get cast as floats when transposed, so loop through and reassign as ints
    for rating  in ratings_data:
        rating['movieId'] = int(rating['movieId'])
        rating['userId'] = int(rating['userId'])

        userid = rating['userId']
        movie_rating_list = rating_dict.setdefault(userid, [])
        movie_rating_list.append(rating)

    return rating_dict



if __name__ == "__main__":

    if not os.path.isdir(MOVIE_LENS_DIR):
        print 'starting download'
        r = requests.get('http://files.grouplens.org/datasets/movielens/ml-latest-small.zip', stream=True)
        z = zipfile.ZipFile(StringIO.StringIO(r.content))
        print 'starting extract'
        z.extractall()

    if not os.path.isdir(DATA_DIR):
        print 'creating directory', DATA_DIR
        os.makedirs(DATA_DIR)


    print 'running post process data'
    movies = get_movies_data()
    print os.path.join(DATA_DIR, 'movies.json')
    with open(os.path.join(DATA_DIR, 'movies.json'), 'wb') as f:
        json.dump(movies,  f, indent=4, separators=(',', ': '))

    ratings = get_ratings_data()
    with open(os.path.join(DATA_DIR, 'ratings.json'), 'wb') as f:
        json.dump(ratings, f, indent=4, separators=(',', ': '))



