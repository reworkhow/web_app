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

user = 'haocheng' #add your username here (same as previous postgreSQL)
host = 'localhost'
dbname = 'ethopia_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Good Afternoon' },
       )

@app.route('/mytest')
def mytest():
    return render_template("multi_select_test.html",
       title = 'Home', user = { 'nickname': 'Good Afternoon' },
       )

@app.route('/mytest_output')
def mytest_output():
    hao1 = request.form.getlist('mymultiselect123')
    print hao1
    return render_template("multi_select_test_output.html",
       title = hao1, user = { 'nickname': hao1 },
       )


@app.route('/db')
def birth_page():
    sql_query = """
                SELECT * FROM ethopia_data_table WHERE "Stillbirth_rate" > 1.0;
                """
    query_results = pd.read_sql_query(sql_query,con)
    births = ""
    for i in range(0,10):
        births += str(query_results.iloc[i]['Stillbirth_rate'])
        births += "<br>"
    return births

@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = """
               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results=pd.read_sql_query(sql_query,con)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    return render_template('cesareans.html',births=births)

@app.route('/input')
def cesareans_input():
    return render_template("input.html")

#@app.route('/output')
#def cesareans_output():
#    return render_template("output.html")

@app.route('/output')
def cesareans_output():
  #pull 'birth_month' from input field and store it
  patient = request.args.get('birth_month')
    #just select the Cesareans  from the birth dtabase for the month that the user inputs

  query = """
  SELECT * FROM ethopia_data_table
  """ % patient
  query_results=pd.read_sql_query(query,con)
  print query_results
  births = []
  for i in range(0,query_results.shape[0]):
      births.append(dict(index=query_results.iloc[i]['malaria rate'], attendant=query_results.iloc[i]['Proportion_of_births_attended_by_skilled_health_personnel'], birth_month=query_results.iloc[i]['Stillbirth_rate']))
      the_result = ''
  the_result = 10
  return render_template("output.html", births = births, the_result = the_result)

@app.route('/test_input')
def test_input():
    return render_template("test_input.html")

@app.route('/test_output')
def test():
  #pull 'birth_month' from input field and store it
  patient = request.args.get('birth_month')
    #just select the Cesareans  from the birth dtabase for the month that the user inputs

  #query = """
  #SELECT * FROM ethopia_data_table WHERE "%s" > 1.0
  #""" % patient
  query = """
  SELECT * FROM ethopia_data_table
  """
  query_results=pd.read_sql_query(query,con)
  print query_results

  #plot
  response='malaria_rate'
  df = query_results
  y  = df.loc[:,response].values
  X =  df.drop(response, axis=1).values

  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)

  feat_labels = df.columns[:]
  forest = RandomForestRegressor(n_estimators=100,
                                 random_state=0,
                                 n_jobs=-1)

  forest.fit(X_train, y_train)
  importances = forest.feature_importances_

  indices = np.argsort(importances)[::-1]

  for f in range(X_train.shape[1]):
     print("%2d) %-*s %f" % (f + 1, 30,
                             feat_labels[indices[f]],
                             importances[indices[f]]))

  img = StringIO.StringIO()
  """
  plt.title('Feature Importances')
  plt.bar(range(X_train.shape[1]),
          importances[indices],
          color='lightblue',
          align='center')

  plt.xticks(range(X_train.shape[1]),
             feat_labels[indices], rotation=90)
  plt.xlim([-1, X_train.shape[1]])
  plt.savefig(img, format='png')
  """
  sns.set(style="whitegrid")

  # Initialize the matplotlib figure
  f, ax = plt.subplots(figsize=(20, 5))

  # Plot the total crashes
  sns.set_color_codes("pastel")
  sns.barplot(x=importances[indices],y=feat_labels[indices].values,
                color="b")

  # Add a legend and informative axis label
  ax.legend(ncol=2, loc="lower right", frameon=True)
  ax.set(xlim=(0, importances[indices][0]+0.1), ylabel="",
        xlabel="Feature Importance")
  sns.despine(left=True, bottom=True)
  plt.savefig(img, format='png')

  img.seek(0)

  plot_url = base64.b64encode(img.getvalue())

  the_result=np.corrcoef(y_test,forest.predict(X_test))[0,1]
  return render_template("test_output.html", the_result = the_result,plot_url=plot_url)

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
