import flask
import pickle
import pandas as pd

with open(f'Notebooks/nbamodel3.pkl', 'rb') as f:
    classifier = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def home():
        return(flask.render_template('index.html'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if flask.request.method == 'GET':
        return(flask.render_template('about-us.html'))

    if flask.request.method == 'POST':

        Team = flask.request.form['Team']
        Home_or_Away = flask.request.form['Home_or_Away']
        Opponent = flask.request.form['Opponent']
        Field_Goals_Percentage = flask.request.form['FieldGoals.']
        Three_Point_Shots_Percentage = flask.request.form['X3PointShots.']
        Opponent_Field_Goals_Percentage = flask.request.form['Opp.FieldGoals.']
        Opponent_Three_Point_Shots_Percentage = flask.request.form['Opp.3PointShots.']

        input_variables = pd.DataFrame([[Team, Home_or_Away,
                                         Opponent, Field_Goals_Percentage,
                                         Three_Point_Shots_Percentage, Opponent_Field_Goals_Percentage, Opponent_Three_Point_Shots_Percentage]],
                                       columns=['Team', 'Home', 'Opponent', 'FieldGoals.',
                                                'X3PointShots.', 'Opp.FieldGoals.', 'Opp.3PointShots.'],
    
                                       dtype=float)

        prediction = classifier.predict(input_variables)[0]

        return flask.render_template('about-us.html',
                                     original_input=
                                     {'Team': Team,
                                      'Home_or_Away': Home_or_Away,
                                      'Opponent': Opponent,
                                      'Field_Goals_Percentage': Field_Goals_Percentage,
                                      'Three_Point_Shots_Percentage': Three_Point_Shots_Percentage,
                                      'Opponent_Field_Goals_Percentage': Opponent_Field_Goals_Percentage,
                                      'Opponent_Three_Point_Shots_Percentage': Opponent_Three_Point_Shots_Percentage},
                                     result=prediction)
@app.route('/aboutus')
def aboutus():
    return(flask.render_template('elements.html'))


if __name__ == '__main__':
    app.run()
