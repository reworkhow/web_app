from flask import request
from flask import render_template
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import StringIO
import base64
from a_Model import ModelIt
from random_forest import plot
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import json
import plotly.plotly
import seaborn as sns
from sklearn.preprocessing import Imputer
import pickle
import HTMLParser


#user = 'haocheng' #add your username here (same as previous postgreSQL)
#host = 'localhost'
#dbname = 'ethopia_db'
#db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
#con = None
#con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/index')
def index123():
    return render_template("index.html")

@app.route('/features')
def features():
    interventions = request.args.getlist('interventions')#request.form.get also not work
    outcome       = request.args.getlist('outcome')#request.form.get also not work

    """
    interventions=['Number of staff', 'Number of Doctors',
       'Percentage of underweight Children',
       'Proportion of children 6-59 months with severe acute malnutrition',
       'Proportion of HHs using latrine',
       'Proportion of  households covered with mosquito bed nets(LLIN)',
       'Proportion of functional 1 to 5 networks',
       'Health budget utilization', 'Essential drugs availability',
       'Integrated Supportive supervision', 'Reporting completeness']
       """

    df = pd.read_csv("/Users/haocheng/Desktop/web_app2/flaskexample/data/data.csv")
    #weight
    w1 = df['Reporting completeness'].values

    y  = df.loc[:,'Malaria rate'].values
    df = df.drop('Malaria rate',axis=1)
    df = df.loc[:,interventions]

    #imputaion
    #imr = Imputer(missing_values='NaN', strategy='median', axis=0)
    #imr = imr.fit(df)
    #X   = imr.transform(df.values)

    #feat_labels = df.columns[:]
    feat_labels = df.loc[:,interventions].columns[:]

    X_train, y_train = df.values ,y

    forest = RandomForestRegressor(n_estimators=1000,
                                   random_state=999,
                                   n_jobs=-1,max_depth=5)

    forest.fit(X_train, y_train,sample_weight=w1)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    pickle.dump(forest, open( "/Users/haocheng/Desktop/web_app2/flaskexample/data/forest.p", "wb" ) )
    pickle.dump(df, open( "/Users/haocheng/Desktop/web_app2/flaskexample/data/df.p", "wb" ) )

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[indices[f]],
                                importances[indices[f]]))

    return render_template("feature_importance.html",
    Feature_1=feat_labels[indices[0]],Feature_2=feat_labels[indices[1]],
    Feature_3=feat_labels[indices[2]],Feature_4=feat_labels[indices[3]],
    Feature_5=feat_labels[indices[4]],
    Importance_1=int(100*importances[indices[0]]),Importance_2=int(100*importances[indices[1]]),
    Importance_3=int(100*importances[indices[2]]),Importance_4=int(100*importances[indices[3]]),
    Importance_5=int(100*importances[indices[4]]),)

#@app.route('/map')
#def index10():
#    return render_template("mapbox.html")


@app.route('/map')
def map():
    map_intervention = request.args.getlist('map_intervention')[0]
    map_county       = request.args.getlist('map_county')[0]
    map_input        = request.args.getlist('map_input')[0]
    print map_intervention,map_county,map_input

    forest=pickle.load(open("/Users/haocheng/Desktop/web_app2/flaskexample/data/forest.p", "rb" ) )
    df    =pickle.load(open("/Users/haocheng/Desktop/web_app2/flaskexample/data/df.p", "rb" ) )

    X_old = df.values.copy()
    df[map_intervention]=df[map_intervention]+int(map_input)
    X_new = df.values.copy()
    #overall add same value
    myafter_all = forest.predict(X_new)
    mybefore_all = forest.predict(X_old)

    #return whole heatmap
    data = pd.read_csv("/Users/haocheng/Desktop/web_app2/flaskexample/data/data.csv")
    mymap = data.loc[:,['lat','lng']]
    mymap['Malaria positivity rate (Calculated Indicators)']=myafter_all
    #mymap['Malaria positivity rate (Calculated Indicators)']=data.loc[:,'Malaria positivity rate (Calculated Indicators)']
    mymap['Malaria positivity rate (Calculated Indicators)']=4.0

    #return for one county
    county_index = data.Location.values.tolist().index(map_county)
    myafter = 2*mybefore_all[county_index]-myafter_all[county_index]
    mybefore = mybefore_all[county_index]

    #send json
    test=mymap.T.to_dict().values()
    json_string = json.dumps(test)
    print json_string
    #parser = HTMLParser.HTMLParser()
    #json_string = parser.unescape(json_string)


    return render_template("map.html",before=mybefore,after=myafter,json_string=json_string)


@app.route('/plot')
def plotly_test():
    rng = pd.date_range('1/1/2011', periods=7500, freq='H')
    ts = pd.Series(np.random.randn(len(rng)), index=rng)

    graphs = [
        dict(
            data=[
                dict(
                    x=[1, 2, 3],
                    y=[10, 20, 30],
                    type='scatter'
                ),
            ],
            layout=dict(
                title='first graph'
            )
        ),

        dict(
            data=[
                dict(
                    x=[1, 3, 5],
                    y=[10, 50, 30],
                    type='bar'
                ),
            ],
            layout=dict(
                title='second graph'
            )
        ),

        dict(
            data=[
                dict(
                    x=ts.index,  # Can use the pandas data structures directly
                    y=ts
                )
            ]
        )
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('plotly.html',
                           ids=ids,
                           graphJSON=graphJSON)
