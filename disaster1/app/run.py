import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    related_counts = df.groupby('related').count()['message']
    related = list(related_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs_related = [
        {
            'data': [
                Bar(
                    x=related,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message related',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    related_ids = ["graph-{}".format(i) for i, _ in enumerate(graphs_related)]
    related_graphJSON = json.dumps(graphs_related, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=related_ids, graphJSON=related_graphJSON)
    #request masssage plot
    request_counts = df.groupby('request').count()['message']
    request = list(request_counts.index)
    graphs_request = [
        {
            'data': [
                Bar(
                    x=request,
                    y=request_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message request',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "request"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    request_ids = ["graph-{}".format(i) for i, _ in enumerate(graphs_request)]
    request_graphJSON = json.dumps(graphs_request, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=request_ids, graphJSON=request_graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()