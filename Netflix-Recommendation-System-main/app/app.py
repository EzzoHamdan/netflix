from flask import Flask, render_template, request

from recommender import NetflixRecommender


app = Flask(__name__)
recommender = NetflixRecommender('NetflixDataset.csv')


@app.route('/')
def index():
    return render_template(
        'index.html',
        languages=recommender.available_languages,
        titles=recommender.available_titles,
    )


@app.route('/about', methods=['POST'])
def getvalue():
    movienames = request.form.getlist('titles')
    selected_languages = request.form.getlist('languages')

    ranked_df = recommender.recommend(
        seed_titles=movienames,
        selected_languages=selected_languages,
        top_k=60,
    )

    images = ranked_df['Image'].tolist() if not ranked_df.empty else []
    titles = ranked_df['Title'].tolist() if not ranked_df.empty else []
    return render_template('result.html', titles=titles, images=images)


@app.route('/moviepage/<name>')
def movie_details(name):
    details = recommender.get_movie_details(name)
    return render_template('moviepage.html', details=details)


if __name__ == '__main__':
    app.run(debug=False)
