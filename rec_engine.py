import json
import config
import os
from collections import Counter
import numpy as np


movies_data = json.load(open(os.path.join(config.BASE_DIR, 'data', 'movies.json'), 'rb'))
ratings_data = json.load(open(os.path.join(config.BASE_DIR, 'data', 'ratings.json'), 'rb'))

NUM_USERS = len(ratings_data)
NUM_MOVIES = len(movies_data)

#normalize each movie ID to a value between 0 and NUM_MOVIES - 1
MOVIE_TO_IDX_LOOKUP = dict(zip(map(int, movies_data.keys()), range(NUM_MOVIES)))
IDX_TO_MOVIE_LOOKUP = dict([(v, k) for k,v in MOVIE_TO_IDX_LOOKUP.items()])

def get_all_ratings():
    """
    Loop through all users and gather ratings list into 1 large table (list of dicts)
    :return:
    """
    all_ratings = []
    for ratings in ratings_data.values():
        all_ratings.extend(ratings)
    return all_ratings


def get_user_summaries():
    """
    Summary report for all users
    :return:
    """
    summary_data = {}
    for user_id in ratings_data:
        data = summary_data.setdefault(user_id, [0.0, 0.0])
        for rating in ratings_data[user_id]:
            data[0] += rating['rating']
            data[1] += 1.0
    return summary_data


def get_summary_by_user(user_id):
    str_user_id = str(user_id) #json all keys are strings
    rating_list = ratings_data.get(str_user_id, [])
    total_rating = 0.0
    total_rating_ct = len(rating_list)

    genre_rating = {}

    sorted_ratings = []

    for rating in rating_list:
        score = rating['rating']

        total_rating += score
        movie_id = rating['movieId']
        movie = movies_data.get(str(movie_id), {})
        rating['title'] = movie['title']
        rating['genres'] = movie['genres']
        sorted_ratings.append(rating)
        for genre in movie.get('genres', []):
            summary_data = genre_rating.setdefault(genre, [0.0, 0.0])
            summary_data[0] += score
            summary_data[1] += 1.0

    sorted_ratings.sort(key = lambda d : -d['rating'])

    return {
        'genre_rating': genre_rating,
        'total_rating': [total_rating, total_rating_ct],
        'sorted_rating': sorted_ratings,
        'average_rating': total_rating/total_rating_ct,
    }


def print_performance_summary(user_id):
    summary = get_summary_by_user(user_id)

    print 'average rating: ', summary['total_rating'][0] / summary['total_rating'][1], summary['total_rating'][1]
    print
    print 'genre_summary: '
    for genre, values in sorted(summary['genre_rating'].items(), key = lambda (k,v) : -(v[0]/v[1]))[:25]:
        print '%s - %.2f'%(genre, values[0]/values[1]), values[1]

    print

    print 'sorted_rating: '

    for rating in summary['sorted_rating'][: 25]:
        print rating['movieId'], rating['rating'], rating['title']


def compare_users(uid1, uid2):
    summary1 = get_summary_by_user(uid1)
    summary2 = get_summary_by_user(uid2)



def cosine_similarity(vec1, vec2):

    A = (vec1 * vec2).sum()
    B = np.sqrt(np.sum(np.square(vec1))) * np.sqrt(np.sum(np.square(vec2)))
    return A/B


def get_most_popular_from_list_of_recs(ratings = None, n = 25):
    """
    when you have no data about the user or item, this bases recommendation on popularity alone

    additional improvement would be to downweight older reviews and give stronger weight to newer reviews.

    :return: Counter, keys are movieId and values are "votes"
    """

    if ratings is None:
        ratings = get_all_ratings()

    score_counter = Counter()
    rec_counter = Counter()
    total_rating = Counter()
    for rating in ratings:
        movie_id = rating['movieId']
        score = rating['rating']

        rec_counter[movie_id] += 1.0
        total_rating[movie_id] += score

        if score < 3:
            score_counter[movie_id] += -1.0
        elif score > 3:
            score_counter[movie_id] += 1.0
        else:
            score_counter[movie_id] += 0.5 #give a weak positive signal for ranking it at all

    results = []
    for movie_id, score in score_counter.most_common(n = n):
        data = {
            'movie_id':movie_id,
            'score': score,
            'avg_rating': total_rating[movie_id] / rec_counter[movie_id],
            'rating_ct' : rec_counter[movie_id],
            'title': movies_data[str(movie_id)]['title']
        }
        results.append(data)

    return results


def build_sim_matrix():
    #NOTE: In pracitce NUM_USERS >> NUM_ITEMS and this would be impractical, also NUM_USERS is constantly getting updated

    """

    :return:
    """

    print 'Building Sim Matrix'

    sim_matrix = np.zeros((NUM_USERS, NUM_MOVIES))
    eps = 0.1

    for uid in range(1, NUM_USERS+1):
        user_idx = uid - 1
        user_summary = get_summary_by_user(uid)

        for rating in user_summary['sorted_rating']:
            movie_id = rating['movieId']
            movie_idx = MOVIE_TO_IDX_LOOKUP[movie_id]
            # for those where the user has rated add in the actual rating, we add a small of the avg so that the values
            # with votes arent zeroed out when average is subtracted
            sim_matrix[user_idx][movie_idx] = (rating['rating'] - user_summary['average_rating'] + eps)
    return sim_matrix

def build_cosine_matrix(sim_matrix):

    print 'Building Cosine Matrix'

    A = sim_matrix.dot(sim_matrix.T) # rank NUM_USERS x NUM_USERS

    l2 = (((sim_matrix**2).sum(axis=1))**0.5).reshape((-1, 1)) #rank NUM_USERS x 1, computes the l2 for each user ratings
    B = l2.dot(l2.T) # rank NUM_USERS x NUM_USERS

    cos_mat = A/B
    np.fill_diagonal(cos_mat, 0.0)

    return cos_mat


def build_cofreq_matrix(sim_matrix):
    """
    calculates the number of "non-trivial" reviews that 2 user have in common
    :param sim_matrix:
    :return:
    """

    print 'Building Cofreq Matrix'

    bit_mat = (np.abs(sim_matrix) > 0).astype(int) #this is a matrix of ones when there is a vote and zeroes everywhere else
    return bit_mat.dot(bit_mat.T)



class RecommendationEngine(object):

    sim_mat = None
    bool_mat = None
    cos_mat = None
    cofreq_mat = None

    def __init__(self):
        if self.sim_mat is None:
            self._build_matrices()

    @classmethod
    def _build_matrices(cls):
        cls.sim_mat = build_sim_matrix()
        cls.bool_mat = (np.abs(cls.sim_mat) > 0).astype(int)
        cls.cos_mat = build_cosine_matrix(cls.sim_mat)
        cls.cofreq_mat = build_cofreq_matrix(cls.sim_mat)

class UserBasedRecommendation(RecommendationEngine):

    def __init__(self, userid, min_recs_in_common = 10, max_sim_users = 10):
        self.userid = userid
        self.user_sim_row_idx = userid - 1
        self.max_sim_users = max_sim_users
        self.min_recs_in_common = min_recs_in_common

        super(UserBasedRecommendation, self).__init__()

        self.cos_sim_vec = self.cos_mat[self.user_sim_row_idx]
        self.cofreq_vec = self.cofreq_mat[self.user_sim_row_idx]
        self.has_min_recs_in_common = (self.cofreq_vec >= self.min_recs_in_common)

    def get_most_sim_users(self, reverse=False):

        sort_dir = 1 if reverse else -1
        sorted_sim = np.argsort(sort_dir * self.cos_sim_vec * self.has_min_recs_in_common)

        results = [{'user_id': (idx + 1),
                    'recs_in_common': self.cofreq_vec[idx],
                    'sim_score': self.cos_sim_vec[idx] }
                        for idx in sorted_sim[:self.max_sim_users]]
        return results

    def get_most_popular_items_from_sim_users(self, n =  25):

        most_sim_users = self.get_most_sim_users()

        ratings = [] #loop through users, get summaries

        this_users_summary = get_summary_by_user(self.userid)
        seen_movies = set([rating['movieId'] for rating in this_users_summary['sorted_rating']])

        movie_ctr = Counter()


        for user in most_sim_users:
            user_id = user['user_id']
            user_summary = get_summary_by_user(user_id)
            user_top_ratings = [rating for rating in user_summary['sorted_rating'] if rating['rating'] > 3.49] #get 4 and 5 ratings for user
            for rating in user_top_ratings:
                if rating['movieId'] not in seen_movies:
                    movie_ctr[rating['movieId']] += 1

        recommendations = []

        for movie_id, votes in movie_ctr.most_common( n = 25 ):
            movie_dict = movies_data[str(movie_id)].copy()
            movie_dict['votes'] = votes
            recommendations.append( movie_dict )

        return recommendations


class MovieBasedRecommendation(RecommendationEngine):

    def __init__(self, movie_id, max_sim_results = 10, min_users_in_common = 10):

        self.min_users_in_common = min_users_in_common
        self.max_sim_results = max_sim_results
        self.movie_id = int(movie_id)
        self.movie_sim_mat_col_idx = MOVIE_TO_IDX_LOOKUP[self.movie_id]

        super(MovieBasedRecommendation, self).__init__()
        self.movie_sim_vec = self.sim_mat[:, self.movie_sim_mat_col_idx]
        self.review_ct = self.bool_mat[:, self.movie_sim_mat_col_idx].sum()

    def get_most_similar_movies(self, reverse=False):

        sim_arr = np.zeros((NUM_MOVIES,))
        for movie_idx in range(self.sim_mat.shape[1]):
            movie_vec = self.sim_mat[:, movie_idx]
            cos_sim = cosine_similarity(self.movie_sim_vec, movie_vec)
            if movie_idx != self.movie_sim_mat_col_idx:
                sim_arr[movie_idx] = cos_sim

        sort_dir = 1 if reverse else -1
        sorted_movies = np.argsort(sort_dir * sim_arr)

        rating_results = []
        for movie_idx in sorted_movies:
            movie_id = IDX_TO_MOVIE_LOOKUP[movie_idx]
            ratings_in_common = (self.bool_mat[:, self.movie_sim_mat_col_idx] * self.bool_mat[:, movie_idx]).sum()

            if ratings_in_common >= self.min_users_in_common:
                rating_results.append( (movies_data[str(movie_id)], ratings_in_common, sim_arr[movie_idx]) )

            if len(rating_results) >= self.max_sim_results:
                break


        return rating_results

    def get_most_in_common(self, reverse=False):

        common_arr = np.zeros((NUM_MOVIES,))
        for movie_idx in range(self.sim_mat.shape[1]):

            ratings_in_common = (self.bool_mat[:, self.movie_sim_mat_col_idx] * self.bool_mat[:, movie_idx]).sum()
            if movie_idx != self.movie_sim_mat_col_idx:
                common_arr[movie_idx] = ratings_in_common

        sort_dir = 1 if reverse else -1
        sorted_movies = np.argsort(sort_dir * common_arr)

        rating_results = []
        for movie_idx in sorted_movies[:self.max_sim_results]:
            movie_id = IDX_TO_MOVIE_LOOKUP[movie_idx]

            cos_sim = cosine_similarity(self.sim_mat[:, self.movie_sim_mat_col_idx], self.sim_mat[:, movie_idx])
            rating_results.append( (movies_data[str(movie_id)], common_arr[movie_idx], cos_sim) )


        return rating_results

if __name__ == "__main__":
    pass

