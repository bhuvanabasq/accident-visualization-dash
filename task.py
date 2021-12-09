#Import all the necessary dependables
import os
import dash
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from pandas import read_csv, DataFrame
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


def task():
    """
    For the last assignment, there is only one task, which will use your knowledge from all previous assignments.
    If your methods of a1, a2 and a3 are well implemented, a4 will be fairly simple, so reuse the methods if possible for your own
    benefit! If you prefer, you can reimplement any logic you with in the assignment4 folder.

    For this task, feel free to create new files, modules or methods in this file as you see fit. Our test will be done by calling this
    task() method, and we expect to receive the dash app back (similar to a3) and we will run it. No other method will be called by us, so
    make sure your code is running as expected. We will basically run this code: `task().run_server(debug=True)`

    For this task, you will build a dash app where we can perform a simple form of interactivity on it. We will use the accidents.csv
    dataset. This accidents.csv dataset has information about traffic accidents in the UK, and is available to you now.
    You will show the most accident prone areas in the UK on a map, so the regular value shown in the map should let us know the number of accidents
    that occurred in a specific area of the map (which should include the accident's severity as well as a weight factor). That said, the purpose of
    this map is to be used by the police to identify how they can do things better.

    **This is raw data, so preprocess the data as per requirement. Drop columns that you feel are unnecessary for the analysis or clustering. 
    Don't forget to add comments about why you did what you did**

    
    ##############################################################
    # Your task will be to Implement all the below functionalities
    ##############################################################

    1. (40pts) Implement a map with the GPS points for the accidents per month. Have a slider(#slider1) that can be used to filter accident data for the month I need.
        You are free to choose a map style, but I suggest using a scatter plot map.

    2. (10pts) Implement a dropdown to select few of the numerical columns in the dataset that can be used meaningfully to represent the size of the GPS points. 
        By default the size of the GPS point on map should be based on the value of "accident_severity" column.

    3. (30pts) You have to Cluster the points now. Be sure to have a text somewhere on the webpage that says what clustering algorithm you are using (e.g. KMeans, dbscan, etc).
        For clustering, you should run a clustering method over the dataset (it should run fairly fast for quick iteration, so make sure to use a simple clustering procedure)
        **COLOR** the GPS points based on the clusetring label returned from the algorithm.

    4. (10pts) Have a button(#run_clustering) to run or rerun the clustering algorithm over the filtered dataset (filtered using #slider1 to select months).

    5. (10pts) At least one parameter of the clustering algorithm should be made available for us to tinker with as a button/slider/dropdown. 
        When I change it and click #run_clustering button, your code should rerun the clustering algorithm. 
        example: change the number of clusters in kmeans or eps value in dbscan.

        Please note: distance values should be in meters, for example dbscan uses eps as a parameter. This value should be read in mts from users and converted appropriately to be used in clustering, 
        so input_eps=100 should mean algorithm uses 100mts circle for finding core and non-core points. 
  
    The total points is 100pts
    """
    FONT_FAMILY = "Arial"

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    acc = read_csv('accidents.csv', index_col=0).dropna(how='any', axis=0)

    # type casting date string to datetime object for easy querying
    acc['date'] = pd.to_datetime(acc['date'])
    acc['accident_reference'] = acc['accident_reference'].astype(str)

    # choosing few numerical columns for the sake of manipulating GPS marker size and normalising them for reuse
    data_size_columns = ['accident_severity', 'number_of_casualties', 'number_of_vehicles', 'speed_limit']
    for col in data_size_columns:
        norm = MinMaxScaler().fit(acc[col].values.reshape(-1, 1))
        # transform training data
        acc['rescaled_'+col] = norm.transform(acc[col].values.reshape(-1, 1))

    app.layout = html.Div([
        html.H1(
            'Traffic Accidents in the UK',
            style={
                'paddingLeft': 50,
                'fontFamily': FONT_FAMILY
            }
        ),
        html.Hr(),
        html.Div([  # Holds the widgets & Descriptions

            html.Div([

                html.H3(
                    '''In 2020, the UK suffered {:,} traffic accidents'''.format(len(acc)),
                    style={
                        'fontFamily': FONT_FAMILY
                    }
                ),
                html.Div(
                    '''Clustering algo used: K-Means''',
                ),
                html.Div(
                    '''Select the month of the accident:''',
                    style={
                        'paddingTop': 20,
                        'paddingBottom': 10
                    }
                ),

                html.Div(
                    dcc.Slider(
                        id='month-slider',
                        min=1,
                        max=12,
                        step=1,
                        value=1,
                        marks={
                            1: 'Jan',
                            2: 'Feb',
                            3: 'Mar',
                            4: 'Apr',
                            5: 'May',
                            6: 'Jun',
                            7: 'Jul',
                            8: 'Aug',
                            9: 'Sep',
                            10: 'Oct',
                            11: 'Nov',
                            12: 'Dec',
                        },
                    ),
                ),
                html.Div(
                    '''Select the type with which we show the size of point:''',
                    style={
                        'paddingTop': 20,
                        'paddingBottom': 10
                    }
                ),
                dcc.Dropdown(
                    id='size-dropdown',
                    options=[
                        {'label': 'Severity', 'value': 'accident_severity'},
                        {'label': 'Number of Casualties', 'value': 'number_of_casualties'},
                        {'label': 'Vehicles Involved', 'value': 'number_of_vehicles'},
                        {'label': 'Speed Limit', 'value': 'speed_limit'},
                    ],
                    value='accident_severity'
                ),

                html.Div(
                    '''Select number of clusters for k-means:''',
                    style={
                        'paddingTop': 20,
                        'paddingBottom': 10
                    }
                ),

                html.Div(
                    dcc.Slider(
                        id='cluster-slider',
                        min=2,
                        max=10,
                        step=1,
                        value=2,
                        marks={
                            2: '2',
                            3: '3',
                            4: '4',
                            5: '5',
                            6: '6',
                            7: '7',
                            8: '8',
                            9: '9',
                            10: '10',
                        },
                    ),
                ),
                html.Button('Run Clustering', id='submit'),

            ],
                style={
                    "width": '40%',
                    'display': 'inline-block',
                    'paddingLeft': 50,
                    'paddingRight': 10,
                    'boxSizing': 'border-box',
                }
            ),

            html.Div([  # Holds the map & the widgets

                dcc.Graph(id="map")  # Holds the map in a div to apply styling to it

            ],
                style={
                    "width": '60%',
                    'float': 'right',
                    'display': 'inline-block',
                    'paddingRight': 5,
                    'paddingLeft': 5,
                    'boxSizing': 'border-box',
                    'fontFamily': FONT_FAMILY
                })

        ],
            style={'paddingBottom': 20}),

    ])

    @app.callback(
        [Output('map', 'figure')],
        [Input('month-slider', 'value'),
         Input('size-dropdown', 'value'),
         Input('submit', 'n_clicks')],
        [State('cluster-slider', 'value')]
    )
    def update_map(slider, size, n_clicks, n_cluster):

        # filtering using month
        rows_by_month = acc.loc[acc['date'].dt.month == slider]

        # do k-means clustering
        # using only three columns to cluster, can add more numerical columns but scope was not defined in question
        df1 = rows_by_month[['longitude', 'latitude', "police_force"]]

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_cluster).fit(df1)

        fig = go.Figure(data=go.Scattergeo(
            lon=rows_by_month['longitude'],
            lat=rows_by_month['latitude'],
            text=rows_by_month['rescaled_'+size],
            mode='markers',
            marker={
                # using normalised values and multiplying by 10 to convert from 0-1 range to 1-10 range,
                # 0.2 added for visual scaling
                'size': (rows_by_month['rescaled_'+size]+0.2)*10,
                'color': km.labels_,
            },

        ))

        fig.update_layout(
            height=600,
            width=800
        )
        # code to centre map around UK
        fig.update_geos(
            center=dict(lon=-2, lat=54.5),
            lataxis_range=[42, 65], lonaxis_range=[-8, 10]
        )
        return [fig]

    return app


app = task()
app.run_server(debug=True)
