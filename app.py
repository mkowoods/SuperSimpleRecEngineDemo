
from flask import Flask, render_template
import rec_engine as rec
app = Flask(__name__)




@app.route('/')
def index():
    summary_items = rec.get_user_summaries().items()
    summary_items.sort(key = lambda x : -int(x[1][1]))

    top_rated_movies = rec.get_most_popular_from_list_of_recs(n=250)


    return render_template('index.html', user_summaries = summary_items, top_rated_movies = top_rated_movies)


@app.route('/user/<int:user_id>')
def user(user_id):

    user_summary = rec.get_summary_by_user(user_id)
    genre_table = sorted([(k, v[1], v[0]/v[1]) for k,v in user_summary['genre_rating'].items()], key=lambda x : -x[1])
    user = rec.UserBasedRecommendation(user_id, max_sim_users=5)

    return render_template('user.html', user_id = user_id,
                                        genre_table = genre_table,
                                        user_summary = user_summary,
                                        sim_users = user.get_most_sim_users(),
                                        dis_sim_users=user.get_most_sim_users(reverse=True),
                                        user_recommendations = user.get_most_popular_items_from_sim_users()
                           )

@app.route('/movie/<int:movie_id>')
def movie(movie_id):
    movie = rec.MovieBasedRecommendation(movie_id, max_sim_results=25)
    #print movie.get_most_similar_movies()
    return render_template('movie.html', movie_id = movie_id,
                                         review_ct = movie.review_ct,
                                         movie_data = rec.movies_data[str(movie_id)],
                                         recs = movie.get_most_similar_movies(),
                                         anti_recs=movie.get_most_similar_movies(reverse=True),
                                         most_in_common = movie.get_most_in_common()
                           )


if __name__ == '__main__':
    app.run(debug=True)
