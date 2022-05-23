import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import gunicorn
import inflect
import joblib
import pandas as pd
import plotly.graph_objects as go

"""Create and configure an instance of the Dash application."""

# Load training dataframe to get wrangled songs
df_train = pd.read_csv('data/df_train.zip', compression='zip')

# Load unwrangled dataset to match the song.
df_rec_lookup = pd.read_csv('data/df_rec_lookup.zip', compression='zip')

# Load pickled model and recommendations lookup dataframe
knn_loader = joblib.load('ml/knn_model.joblib')

logo_link = 'assets/Spotify_Logo_RGB_Green-768x231.png'

# Create the dash app
app = dash.Dash(__name__)
server = app.server
app.title = "Spotify Recommendation App"


# Function to add styling
def style_c():
    layout_style = {'display': 'inline-block', 'margin': '0 auto',
                    'padding': '20px'}
    return layout_style


# Function to generate radar graph for song selected by user
def user_song_fig(df, i):
    # Generate user song and artist names
    user_song_input = df.columns[i]
    user_song_name = df_rec_lookup.loc[user_song_input, 'name']
    user_song_artist = df_rec_lookup.loc[user_song_input, 'artists']. \
        replace("'", "").strip("[]")
    fig_user_song = go.Figure()
    fig_user_song.add_trace(go.Scatterpolar(r=df.iloc[:, i],
                                            theta=df.iloc[:, 0],
                                            fill='toself'))
    fig_user_song.update_layout(polar=dict(radialaxis=dict(visible=True,
                                                           range=[0, 1])),
                                title={'text': f'You have selected <br>'
                                               f'{user_song_name} <br> by '
                                               f'{user_song_artist}',
                                       'xanchor': 'center', 'yanchor': 'top',
                                       'y': 0.95, 'x': 0.5}, showlegend=False)
    return fig_user_song


# Function to generate overlay radar graph for songs recommended by model
def rec_song_fig(df, i):
    # Generate recommended songs and artists names
    rec_song_i_input = df.columns[i]
    rec_song_i_name = df_rec_lookup.loc[rec_song_i_input, 'name']
    rec_song_i_artist = df_rec_lookup.loc[rec_song_i_input, 'artists']. \
        replace("'", "").strip("[]")
    p = inflect.engine()

    fig_rec_song_i = go.Figure()
    # Generate radar graph for the user selected song
    fig_rec_song_i.add_trace(go.Scatterpolar(
        r=df.iloc[:, 1],
        theta=df.iloc[:, 0],
        fill='toself',
        name='User Selected Song'
    ))
    # Generate the overlay radar graph for each of the recommended songs
    fig_rec_song_i.add_trace(go.Scatterpolar(
        r=df.iloc[:, i],
        theta=df.iloc[:, 0],
        fill='toself',
        name=f'{p.ordinal(i - 1)} Recommended Song'
    ))
    # Generate graph title and legend
    fig_rec_song_i.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title={'text': f'Your {p.ordinal(i - 1)} recommended song is <br>'
                       f'{rec_song_i_name} <br> by {rec_song_i_artist}',
               'xanchor': 'center', 'yanchor': 'top', 'y': 0.95, 'x': 0.5},
        showlegend=True
    )
    return fig_rec_song_i


# Create base path for mini-Spotify-player
player_base_path = "https://open.spotify.com/embed/track/"

app.layout = html.Div([
    html.Div(
        children=[
            html.Img(src=logo_link, style={
                'width': '25%',
                'height': '25%',
                'margin': '30px 0px 0px 30px'
            }),
            html.H1('Spotify Song Recommendation App'),
        ],
        style={'textAlign': 'center',
               'display': 'inline-block',
               'width': '100%'}),
    html.Div(
        children=[
            html.Div(
                children=[
                    html.H2('Song Selector', style=style_c()),
                    html.H3(
                        'Select the Song for which You Want a Recommendation'
                        ' (from 0 to 1,126,175)',
                    ),
                    html.Br(),
                    dcc.Input(
                        id='user_song',
                        type='number',
                        min=0,
                        max=1126175,
                        placeholder='Enter the Song Number',
                        step=1,
                        value=0,
                        debounce=True,
                        style={'width': '300px', 'height': '30px'}),
                ],
                style={
                    'width': '400px',
                    'height': '400px',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'textAlign': 'center',
                    'border': '1px solid black',
                    'padding': '20px',
                }),
            html.Div(
                children=[
                    dcc.Graph(
                        id='user_song_graph')
                ],
                style={'display': 'inline-block',
                       'width': '33%',
                       'padding': 'auto'
                       }),
            html.Div(
                children=[
                    dcc.Markdown(
                        id='user_song_player',
                        dangerously_allow_html=True)
                ], style={'display': 'inline-block',
                          'verticalAlign': 'top',
                          'width': '33%',
                          'padding': '100px auto',
                          'margin': '100 px auto',
                          }),
        ]),
    html.Div(
        children=[
            html.Div(
                children=[
                    dcc.Graph(
                        id='rec_song_1_graph')
                ],
                style={'display': 'inline-block',
                       'width': '50%',
                       'padding': 'auto'
                       }),
            html.Div(
                children=[
                    dcc.Markdown(
                        id='rec_song_1_player',
                        dangerously_allow_html=True)
                ],
                style={'display': 'inline-block',
                       'verticalAlign': 'top',
                       'width': '50%',
                       'padding': '100px auto',
                       'margin': '100 px auto',
                       })
        ]),
    html.Div(
        children=[
            html.Div(
                children=[
                    dcc.Graph(
                        id='rec_song_2_graph')
                ],
                style={'display': 'inline-block',
                       'width': '50%',
                       'padding': 'auto'
                       }),
            html.Div(
                children=[
                    dcc.Markdown(
                        id='rec_song_2_player',
                        dangerously_allow_html=True)
                ],
                style={'display': 'inline-block',
                       'verticalAlign': 'top',
                       'width': '50%',
                       'padding': '100px auto',
                       'margin': '100 px auto',
                       })
        ]),
    html.Div(
        children=[
            html.Div(
                children=[
                    dcc.Graph(
                        id='rec_song_3_graph')
                ],
                style={'display': 'inline-block',
                       'width': '50%',
                       'padding': 'auto'
                       }),
            html.Div(
                children=[
                    dcc.Markdown(
                        id='rec_song_3_player',
                        dangerously_allow_html=True)
                ],
                style={'display': 'inline-block',
                       'verticalAlign': 'top',
                       'width': '50%',
                       'padding': '100px auto',
                       'margin': '100 px auto',
                       })
        ]),
])


@app.callback(
    Output(component_id='user_song_graph', component_property='figure'),
    Output(component_id='user_song_player', component_property='children'),
    Output(component_id='rec_song_1_graph', component_property='figure'),
    Output(component_id='rec_song_1_player', component_property='children'),
    Output(component_id='rec_song_2_graph', component_property='figure'),
    Output(component_id='rec_song_2_player', component_property='children'),
    Output(component_id='rec_song_3_graph', component_property='figure'),
    Output(component_id='rec_song_3_player', component_property='children'),
    Input(component_id='user_song', component_property='value'))
def update_plot(input_song):
    doc = df_train.iloc[[input_song]]

    # Query Using K-Nearest Neighbors Model
    __, neigh_index = knn_loader.kneighbors(doc)

    # Generate dataframe for visualizations
    df_viz = df_train.iloc[neigh_index[0][:4]]
    # Drop age columns as unnecessary for visualization
    df_viz = df_viz[df_viz.columns.drop(list(df_viz.filter(regex='age')))]
    # Transpose dataframe for use in graph generating function
    df_viz_transposed = df_viz.transpose().reset_index()

    song_id = []
    for i in neigh_index[0][0:4]:
        song_id.append(df_rec_lookup['id'][i])

    # Generate graph and mini-Spotify-player for song selected by user
    fig_user_song = user_song_fig(df_viz_transposed, 1)
    user_song = '''<iframe src="https://open.spotify.com/embed/track/''' + \
                song_id[0] + '''" width="450" height="450" frameborder="0" 
                allowtransparency="true" allow="encrypted-media"></iframe>'''

    # Generate graphs and mini-Spotify-players for songs recommended by model
    fig_rec_song_1 = rec_song_fig(df_viz_transposed, 2)
    rec_song_1 = '''<iframe src="https://open.spotify.com/embed/track/''' + \
                 song_id[1] + '''" width="450" height="450" frameborder="0"
                 allowtransparency="true" allow="encrypted-media"></iframe>'''

    fig_rec_song_2 = rec_song_fig(df_viz_transposed, 3)
    rec_song_2 = '''<iframe src="https://open.spotify.com/embed/track/''' + \
                 song_id[2] + '''" width="450" height="450" frameborder="0" 
                 allowtransparency="true" allow="encrypted-media"></iframe>'''

    fig_rec_song_3 = rec_song_fig(df_viz_transposed, 4)
    rec_song_3 = '''<iframe src="https://open.spotify.com/embed/track/''' + \
                 song_id[3] + '''" width="450" height="450" frameborder="0" 
                 allowtransparency="true" allow="encrypted-media"></iframe>'''

    return fig_user_song, user_song, fig_rec_song_1, rec_song_1, fig_rec_song_2, \
           rec_song_2, fig_rec_song_3, rec_song_3


if __name__ == '__main__':
    app.run_server(debug=False)
