import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objects import Pie
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
    related_counts = df.groupby('related').count()['message']
    related = list(related_counts.index)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs=[]
    graphs1={}
    graphs1 =         {
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
                    'title': "related"
                }
            }
        }
    
#request masssage plot
    request_counts = df.groupby('request').count()['message']
    request = list(request_counts.index)
    graphs2={}
    graphs2 =         {
            'data': [
                Pie(
                    labels=request,
                    values=request_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message request',
                'textinfo': 'value',
                'hoverinfo':'label+percent'
            }
        }
    
    graphs.append(graphs1)
    graphs.append(graphs2)
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)
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
    app.run(host='127.0.1.1', port=3001, debug=True)


if __name__ == '__main__':
    main()