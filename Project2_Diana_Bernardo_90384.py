"""
Energy Services - Project 2
Diana Bernardo 90384
"""

# Import libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px


# Import file for the style of dashbboard
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# Import files
df = pd.read_csv('clean.csv') 
df_errors = pd.read_csv('df_errors.csv')
df_errors = df_errors.round(3)
df_errors = df_errors.set_index('MAE', drop=True)
df_errors2 = pd.read_csv('df_errors2.csv')
df_errors2 = df_errors2.round(3)
df_errors2 = df_errors2.set_index('Forecasting Method', drop=True)
df_pred = pd.read_csv('df_pred.csv')


# Calculate daily mean of variables
df_mean = df
df_mean['Date'] = pd.to_datetime(df_mean['Date'], dayfirst=True)
df_mean = df_mean.set_index('Date')
df_mean['YY-MM-DD'] = df_mean.index.date
df_mean = df_mean.reset_index()
df_mean['YY-MM-DD'] = pd.to_datetime(df_mean['YY-MM-DD'], dayfirst=True)
df_mean = df_mean.groupby(['YY-MM-DD']).mean()
df_mean = df_mean.reset_index()
df_mean.rename(columns = {'YY-MM-DD':'Date'}, inplace = True)


# Day type count
s =  df['Day type'].value_counts().reset_index()
s.columns =['Day Type', 'Day Count']
s['Day Count'] = s['Day Count'].div(24).round(0) # Convert hour count to day count
s['Day Count'] = s['Day Count'].astype(int)


# Initialize the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True


# Define the Layout of the App
app.layout = html.Div([
    
    html.Div(html.Img(src=app.get_asset_url('IST_logo.png'), style={
             'padding': 10, 'height': '10%', 'width': '10%', 'marginRight': 'auto'}, title='IST Logo')),

    html.Div([
        html.H1('Energy Services - Project 2', style={'margin-left': 10}),
        html.H2('Civil Building Energy Forecast Tool', style={'margin-left': 10, 'color': '#2F4F4F'}),
        html.H3('Diana Bernardo 90384', style={'margin-left': 10, 'color': '#808080'}),
        html.Img(src=app.get_asset_url('pavilhao_civil.jpg'), style={
                 'height': '20%', 'width': '20%', 'marginRight': 'auto'}, title='Civil Building'),
    ], style={'textAlign': 'center'}),  
    
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw variables', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Metrics comparison', value='tab-3'),
    ]),
    html.Div(id='tabs-content')      
])


# Function to generate table
def generate_table(df, max_rows):
    df_1 = df
    df_1 = df_1.reset_index()
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df_1.columns])] +
        # Body
        [html.Tr([
            html.Td(df_1.iloc[i][col]) for col in df_1.columns
        ]) for i in range(min(len(df_1), max_rows))]
        , style={'marginLeft': 'auto', 'marginRight': 'auto'})


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H5('Select the variable you want to visualize:'),
            dcc.Dropdown(
                id='variables',
                value=0,
                # Variables used in project 1
                options=[{'value': 0, 'label': 'Power (kWh)'},
                         {'value': 1, 'label': 'Temperature (\N{DEGREE SIGN}C)'},
                         {'value': 2, 'label': 'Solar Radiation (W/m\u00b2)'},
                         {'value': 3, 'label': 'Day Type'},
                         {'value': 4, 'label': 'Weekday'},
                         {'value': 5, 'label': 'Hour'},
                          ]
            ),
            html.Div(id="graph_var"),
            ])
          
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.H4('Power Prediction: select the type of forecasting method'),
                dcc.Dropdown(
                    id='forecast_type',
                    value=0,
                    options=[{'value': 0, 'label': 'Linear Regression'},
                             {'value': 1, 'label': 'Support Vector Regressor'},
                             {'value': 2, 'label': 'Decision Tree Regressor'},
                             {'value': 3, 'label': 'Random Forest'},
                             {'value': 4, 'label': 'Random Forest Uniformized Data'},
                             {'value': 5, 'label': 'Gradient Boosting'},
                             {'value': 6, 'label': 'Extreme Gradient Boosting'},
                             {'value': 7, 'label': 'Bootstrapping'},
                             {'value': 8, 'label': 'Neural Networks'},
                              ]
                ),
                html.Div(id='Prediction'),
            ]),
        ])    

    elif tab == 'tab-3':
         return html.Div([
             html.H4('Comparison between forecasting methods: select the type of error'),
             dcc.Dropdown(
                 id='errors',
                 value=0,
                 options=[{'value': 0, 'label': 'MAE'},
                          {'value': 1, 'label': 'MSE'},
                          {'value': 2, 'label': 'RMSE'},
                          {'value': 3, 'label': 'cvRMSE'},]
             ),
             html.Div(id='errors2'),
         ])

            
@app.callback(
    Output("graph_var", "children"),
    [Input("variables", "value")])

def variables(mode):
    if mode == 0:
        return html.Div([
            html.H5('Select the type of graph to represent the Power:'),
            dcc.Dropdown(
                id='power',
                value=0,
                options=[{'value': 0, 'label': 'Power line graph'},
                         {'value': 1, 'label': 'Daily average power day count'},
                          ]
            ),
            html.Div(id="power2"),
            ])

    elif mode == 1:
        return html.Div([
            html.H5('Select the type of graph to represent the Temperature:'),
            dcc.Dropdown(
                id='temp',
                value=0,
                options=[{'value': 0, 'label': 'Temperature line graph'},
                         {'value': 1, 'label': 'Daily average temperature day count'},
                          ]
            ),
            html.Div(id="temp2"),
            ])  

    elif mode == 2:
        return html.Div([
            html.H5('Select the type of graph to represent the Solar Radiation:'),
            dcc.Dropdown(
                id='sol',
                value=0,
                options=[{'value': 0, 'label': 'Solar radiation line graph'},
                         {'value': 1, 'label': 'Daily average solar radiation day count'},
                          ]
            ),
            html.Div(id="sol2"),
            ])  
 
    elif mode == 3:
        return html.Div([
            html.H5('Select the type of graph to represent the Day Type:'),
            dcc.Dropdown(
                id='day_type',
                value=0,
                options=[{'value': 0, 'label': 'Day type line graph'},
                         {'value': 1, 'label': 'Day type count'},
                          ]
            ),
            html.Div(id="day_type2"),
            ])
    
    elif mode == 4:
        return html.Div([
                dcc.Textarea(
                    id='text',
                    value='Weekday:\n0 - Monday\n1 - Tuesday\n2 - Wednesday\n3 - Thursday\n4 - Friday\n5 - Saturday\n6 - Sunday',
                    style={'width': '10%', 'height': 150},
                ), 
               dcc.Graph(
                   id='2019-weekday',
                   figure={
                       'data': [{'x': df['Date'], 'y': df['Weekday'], 'type': 'line', 'name': 'Weekday','line':dict(color='purple')}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Weekday'},'title': 'Weekday from January to March 2019'}
                       }
                   ),  
            ])

    elif mode == 5:
        return html.Div([
               dcc.Graph(
                   id='2019-hour',
                   figure={
                       'data': [{'x': df['Date'], 'y': df['Hour'], 'type': 'line', 'name': 'Hour','line':dict(color='grey')}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Hour'},'title': 'Hours from January to March 2019'}
                       }
                   ),  
            ])
    

@app.callback(
    Output("power2", "children"),
    [Input("power", "value")])

def power(mode):
    if mode == 0:
        return html.Div([
            html.Div(id="graph"),    
               dcc.Graph(
                   id='2019-data-power',
                   figure={
                       'data': [{'x': df['Date'], 'y': df['Power (kWh)'], 'type': 'line', 'name': 'Power','line':dict(color='red')}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Power (kWh)'},'title': 'Electricity Consumption at Civil Building from January to March 2019 (kWh)'}
                       }
                   ),  
               ])  
                               
    elif mode == 1:
        return html.Div([
                dcc.Graph(
                     id='hist_power',
                     figure = px.histogram(df_mean, x='Power (kWh)', nbins=6,
                         title="Daily Average Energy Consumption Day Count at Civil Building from January to March 2019", 
                         opacity=0.5, color_discrete_sequence=['red'])        
                     ),           
               ])    
           

@app.callback(
    Output("temp2", "children"),
    [Input("temp", "value")])

def temp(mode):
    if mode == 0:
        return html.Div([    
                dcc.Graph(
                    id='2019-data-temperature',
                    figure={
                        'data': [{'x': df['Date'], 'y': df['Temperature (C)'], 'type': 'line', 'name': 'Temperature','line':dict(color='green')}],
                        'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Temperature (\N{DEGREE SIGN}C)'},'title': 'Temperature at Civil Building from January to March 2019 (\N{DEGREE SIGN}C)'}
                        }
                    ),    
                ])           

    elif mode == 1:
        return html.Div([
                dcc.Graph(
                     id='hist_temp',
                     figure = px.histogram(df_mean, x='Temperature (C)', labels={
                         "Temperature (C)": "Temperature (\N{DEGREE SIGN}C)"} , nbins=6,
                         title="Daily Average Temperature Day Count at Civil Building from January to March 2019",
                         opacity=0.5, color_discrete_sequence=['green'])        
                     ),             
               ])
    

@app.callback(
    Output("sol2", "children"),
    [Input("sol", "value")])

def sol(mode):
    if mode == 0:
        return html.Div([    
                dcc.Graph(
                    id='2019-data-solar_rad',
                    figure={
                        'data': [{'x': df['Date'], 'y': df['Solar Rad (W/m2)'], 'type': 'line', 'name': 'Solar Radiation'}],
                        'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Solar Radiation (W/m\u00b2)'},'title': 'Solar Radiation at Civil Building from January to March 2019 (W/m\u00b2)'}
                        }
                    ), 
                ])   
            
    elif mode == 1:
        return html.Div([
                dcc.Graph(
                     id='hist_solar_rad',
                     figure = px.histogram(df_mean, x='Solar Rad (W/m2)', labels={
                         "Solar Rad (W/m2)": "Solar Radiation (W/m\u00b2)"} , nbins=6,
                         title="Daily Average Solar Radiation Day Count at Civil Building from January to March 2019",
                         opacity=0.5,)        
                     ),        
               ])  


@app.callback(
    Output("day_type2", "children"),
    [Input("day_type", "value")])

def day_type(mode):
    if mode == 0:
        return html.Div([ 
                dcc.Textarea(
                    id='text',
                    value='Day Type:\n0 - Weekdays during classes seasons\n1 - Holidays and weekends\n2 - Weekdays during exam seasons\n3 - Weekdays during vacation',
                    style={'width': '25%', 'height': 100},
                ),
                dcc.Graph(
                    id='2019-data-day_type',
                    figure={
                        'data': [{'x': df['Date'], 'y': df['Day type'], 'type': 'line', 'name': 'Day Type','line':dict(color='orange')}],
                        'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Day Type'},'title': 'Day Type at Civil Building from January to March 2019'}
                        }
                    ),  
                ])           

    elif mode == 1:
        return html.Div([
                dcc.Textarea(
                    id='text',
                    value='Day Type:\n0 - Weekdays during classes seasons\n1 - Holidays and weekends\n2 - Weekdays during exam seasons\n3 - Weekdays during vacation',
                    style={'width': '25%', 'height': 100},
                ), 
                html.Div([html.Div([dcc.Graph(
                     id='bar',
                     figure = px.bar(s, x='Day Type', y='Day Count',
                         title="Day Type Count at Civil Building from January to March 2019",
                         opacity=0.5, color_discrete_sequence=['orange'])        
                     ),                
               ], style=dict(width='50%', marginLeft= 'auto', marginRight= 'auto')), 
                html.Div([dcc.Graph(
                     id='pie',
                     figure = px.pie(s, values='Day Count', names='Day Type',
                         title="Day Type Count at Civil Building from January to March 2019")        
                     ),                
               ], style=dict(width='50%', marginLeft= 'auto', marginRight= 'auto')), 
            ], style=dict(display='flex', marginLeft= 'auto', marginRight= 'auto'))
                ])
 
    
@app.callback(
    Output("Prediction", "children"),
    [Input("forecast_type", "value")])

def prediction(mode):
    if mode == 0:              
        return html.Div([
               dcc.Graph(
                   id='LR',
                   figure={
                       'data': [{'x': df_pred['Date'], 'y': df_pred['data'], 'type': 'line', 'name': 'Data'},{'x': df_pred['Date'], 'y': df_pred['LR'], 'type': 'line', 'name': 'Prediction'}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Power (kWh)'},'title': 'Power Prediction and Data from January to March 2019 - Linear Regression'}
                       }
                   ), 
               dcc.Graph(
                    id='LR_scatter',
                    figure = px.scatter(df_pred, x="data", y="LR",
                         labels={
                             "data": "Power Data (kWh)",
                             "LR": "Power Prediction (kWh)",
                         },
                        title="Power Prediction vs Data from January to March 2019 - Linear Regression")            
                    ),
            html.Div([
               html.Br(),
               html.Br(),
               html.H4('Linear Regression errors:', style={'textAlign':'center','font-weight': 'bold'}),
               generate_table(df_errors.iloc[[0]],3)
               ]), 
           ])
                
    elif mode == 1:
          return html.Div([
               dcc.Graph(
                   id='SVR',
                   figure={
                       'data': [{'x': df_pred['Date'], 'y': df_pred['data'], 'type': 'line', 'name': 'Data'},{'x': df_pred['Date'], 'y': df_pred['SVR'], 'type': 'line', 'name': 'Prediction'}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Power (kWh)'},'title': 'Power Prediction and Data from January to March 2019 - Support Vector Regression'}
                       }
                   ),               
               dcc.Graph(
                    id='SVR_scatter',
                    figure = px.scatter(df_pred, x="data", y="SVR",
                         labels={
                             "data": "Power Data (kWh)",
                             "SVR": "Power Prediction (kWh)",
                         },
                        title="Power Prediction vs Data from January to March 2019 - Support Vector Regression")                  
                    ),
             html.Div([
                html.Br(),
                html.Br(),
                html.H4('Support Vector Regression errors:', style={'textAlign':'center','font-weight': 'bold'}),
                generate_table(df_errors.iloc[[1]],3)
                ]),             
            ])

    elif mode == 2:
        return html.Div([
               dcc.Graph(
                   id='DT',
                   figure={
                       'data': [{'x': df_pred['Date'], 'y': df_pred['data'], 'type': 'line', 'name': 'Data'},{'x': df_pred['Date'], 'y': df_pred['DT'], 'type': 'line', 'name': 'Prediction'}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Power (kWh)'},'title': 'Power Prediction and Data from January to March 2019 - Decision Tree Regressor'}
                       }
                   ),               
               dcc.Graph(
                    id='DT_scatter',
                    figure = px.scatter(df_pred, x="data", y="DT",
                         labels={
                             "data": "Power Data (kWh)",
                             "DT": "Power Prediction (kWh)",
                         },
                        title="Power Prediction vs Data from January to March 2019 - Decision Tree Regressor")                
                    ),
             html.Div([
                html.Br(),
                html.Br(),
                html.H4('Decision Tree Regressor errors:', style={'textAlign':'center','font-weight': 'bold'}),
                generate_table(df_errors.iloc[[2]],3)
                ]),                               
            ])

    elif mode == 3:
        return html.Div([
               dcc.Graph(
                   id='RF',
                   figure={
                       'data': [{'x': df_pred['Date'], 'y': df_pred['data'], 'type': 'line', 'name': 'Data'},{'x': df_pred['Date'], 'y': df_pred['RF'], 'type': 'line', 'name': 'Prediction'}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Power (kWh)'},'title': 'Power Prediction and Data from January to March 2019 - Random Forest Regressor'}
                       }
                   ),               
               dcc.Graph(
                    id='RF_scatter',
                    figure = px.scatter(df_pred, x="data", y="RF",
                         labels={
                             "data": "Power Data (kWh)",
                             "RF": "Power Prediction (kWh)",
                         },
                        title="Power Prediction vs Data from January to March 2019 - Random Forest Regressor")                 
                    ),
             html.Div([
                html.Br(),
                html.Br(),
                html.H4('Random Forest Regressor errors:', style={'textAlign':'center','font-weight': 'bold'}),
                generate_table(df_errors.iloc[[3]],3)
                ]), 
            ])

    elif mode == 4:
        return html.Div([
               dcc.Graph(
                   id='RF_unif',
                   figure={
                       'data': [{'x': df_pred['Date'], 'y': df_pred['data'], 'type': 'line', 'name': 'Data'},{'x': df_pred['Date'], 'y': df_pred['RF_unif'], 'type': 'line', 'name': 'Prediction'}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Power (kWh)'},'title': 'Power Prediction and Data from January to March 2019 - Random Forest Uniformized Data'}
                       }
                   ),               
               dcc.Graph(
                    id='RF_unif_scatter',
                    figure = px.scatter(df_pred, x="data", y="RF_unif",
                         labels={
                             "data": "Power Data (kWh)",
                             "RF_unif": "Power Prediction (kWh)",
                         },
                        title="Power Prediction vs Data from January to March 2019 - Random Forest Uniformized Data")                 
                    ),
             html.Div([
                html.Br(),
                html.Br(),
                html.H4('Random Forest Uniformized Data errors:', style={'textAlign':'center','font-weight': 'bold'}),
                generate_table(df_errors.iloc[[4]],3)
                ]),            
            ])

    elif mode == 5:
        return html.Div([
               dcc.Graph(
                   id='GB',
                   figure={
                       'data': [{'x': df_pred['Date'], 'y': df_pred['data'], 'type': 'line', 'name': 'Data'},{'x': df_pred['Date'], 'y': df_pred['GB'], 'type': 'line', 'name': 'Prediction'}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Power (kWh)'},'title': 'Power Prediction and Data from January to March 2019 - Gradient Boosting'}
                       }
                   ),
               dcc.Graph(
                    id='GB_scatter',
                    figure = px.scatter(df_pred, x="data", y="GB",
                         labels={
                             "data": "Power Data (kWh)",
                             "GB": "Power Prediction (kWh)",
                         },
                        title="Power Prediction vs Data from January to March 2019 - Gradient Boosting")                 
                    ),
             html.Div([
                html.Br(),
                html.Br(),
                html.H4('Gradient Boosting errors:', style={'textAlign':'center','font-weight': 'bold'}),
                generate_table(df_errors.iloc[[5]],3)
                ]),             
            ])

    elif mode == 6:
        return html.Div([
               dcc.Graph(
                   id='XGB',
                   figure={
                       'data': [{'x': df_pred['Date'], 'y': df_pred['data'], 'type': 'line', 'name': 'Data'},{'x': df_pred['Date'], 'y': df_pred['XGB'], 'type': 'line', 'name': 'Prediction'}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Power (kWh)'},'title': 'Power Prediction and Data from January to March 2019 - Extreme Gradient Boosting'}
                       }
                   ),               
               dcc.Graph(
                    id='XGB_scatter',
                    figure = px.scatter(df_pred, x="data", y="XGB",
                         labels={
                             "data": "Power Data (kWh)",
                             "XGB": "Power Prediction (kWh)",
                         },
                        title="Power Prediction vs Data from January to March 2019 - Extreme Gradient Boosting")                 
                    ),
             html.Div([
                html.Br(),
                html.Br(),
                html.H4('Extreme Gradient Boosting errors:', style={'textAlign':'center','font-weight': 'bold'}),
                generate_table(df_errors.iloc[[6]],3)
                ]), 
            ])   

    elif mode == 7:
        return html.Div([
               dcc.Graph(
                   id='BT',
                   figure={
                       'data': [{'x': df_pred['Date'], 'y': df_pred['data'], 'type': 'line', 'name': 'Data'},{'x': df_pred['Date'], 'y': df_pred['BT'], 'type': 'line', 'name': 'Prediction'}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Power (kWh)'},'title': 'Power Prediction and Data from January to March 2019 - Bootstrapping'}
                       }
                   ),               
               dcc.Graph(
                    id='BT_scatter',
                    figure = px.scatter(df_pred, x="data", y="BT",
                         labels={
                             "data": "Power Data (kWh)",
                             "BT": "Power Prediction (kWh)",
                         },
                        title="Power Prediction vs Data from January to March 2019 - Bootstrapping")                 
                    ),
             html.Div([
                html.Br(),
                html.Br(),
                html.H4('Bootstrapping errors:', style={'textAlign':'center','font-weight': 'bold'}),
                generate_table(df_errors.iloc[[7]],3)
                ]), 
            ]) 

    elif mode == 8:
        return html.Div([
               dcc.Graph(
                   id='NN',
                   figure={
                       'data': [{'x': df_pred['Date'], 'y': df_pred['data'], 'type': 'line', 'name': 'Data'},{'x': df_pred['Date'], 'y': df_pred['NN'], 'type': 'line', 'name': 'Prediction'}],
                       'layout':{'xaxis': {'title':'Date'},'yaxis': {'title':'Power (kWh)'},'title': 'Power Prediction and Data from January to March 2019 - Neural Networks'}
                       }
                   ),               
               dcc.Graph(
                    id='NN_scatter',
                    figure = px.scatter(df_pred, x="data", y="NN",
                         labels={
                             "data": "Power Data (kWh)",
                             "NN": "Power Prediction (kWh)",
                         },
                        title="Power Prediction vs Data from January to March 2019 - Neural Networks")                 
                    ),
             html.Div([
                html.Br(),
                html.Br(),
                html.H4('Neural Networks errors:', style={'textAlign':'center','font-weight': 'bold'}),
                generate_table(df_errors.iloc[[8]],3)
                ]), 
             ])

          
@app.callback(
    Output("errors2", "children"),
    [Input("errors", "value")])

def errors(mode):
    if mode == 0:
        return html.Div([
             html.Div([
                html.Br(),
                html.Br(),
                generate_table(df_errors2.iloc[[0,1,2,3,4,5,6,7,8], [0]],9)
                ], style=dict(width='33%', marginLeft= 'auto', marginRight= 'auto')), 
               ])

    elif mode == 1:
        return html.Div([
             html.Div([
                html.Br(),
                html.Br(),
                generate_table(df_errors2.iloc[[0,1,2,3,4,5,6,7,8], [1]],9)
                ], style=dict(width='33%', marginLeft= 'auto', marginRight= 'auto')), 
               ])

    elif mode == 2:
        return html.Div([
             html.Div([
                html.Br(),
                html.Br(),
                generate_table(df_errors2.iloc[[0,1,2,3,4,5,6,7,8], [2]],9)
                ], style=dict(width='33%', marginLeft= 'auto', marginRight= 'auto')), 
               ])

    elif mode == 3:
        return html.Div([
             html.Div([
                html.Br(),
                html.Br(),
                generate_table(df_errors2.iloc[[0,1,2,3,4,5,6,7,8], [3]],9)
                ], style=dict(width='33%', marginLeft= 'auto', marginRight= 'auto')), 
               ])
  

if __name__ == '__main__':
    app.run_server(debug=True)