import dash
from dash import dcc
from dash import html
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

"""Create and configure an instance of the Dash application."""

# Load training dataframe to get wrangled songs
df_train = pd.read_csv('data/df_train.zip', compression='zip')

# Load unwrangled dataset to match the song.
# df_rec_lookup = pd.read_csv('data/df_rec_lookup.zip', compression='zip')

# Load pickled model and recommendations lookup dataframe
knn_loader = joblib.load('ml/knn_model.joblib')

logo_link = 'https://assets.datacamp.com/production/repositories/5893/datasets/' \
            '2bac9433b0e904735feefa26ca913fba187c0d55/e_com_logo.png'

# Create the dash app
app = dash.Dash()


# Create a function to add styling
def style_c():
    layout_style = {'display': 'inline-block', 'margin': '0 auto',
                    'padding': '20px'}
    return layout_style


# def blank_fig():
#     fig = go.Figure(go.Scatter(x=[], y=[]))
#     fig.update_layout(template=None)
#     fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
#     fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
#
#     return fig


app.layout = html.Div([
    html.Img(src=logo_link, style={'margin': '30px 0px 0px 0px'}),
    html.H1('Spotify Song Recommendation App'),
    html.Div(
        children=[
            html.Div(
                children=[
                    html.H2('Song Selector', style=style_c()),
                    html.H3('Select the Song for which You Want a Recommendation'
                            ' (from 0 to 1,126,175)'),
                    html.Br(),
                    dcc.Input(id='user_song', type='number', min=0, max=1126175,
                              placeholder='Enter the Song Number', step=1,
                              debounce=True, style={'width': '300px',
                                                    'height': '30px'}),
                ],
                style={'width': '350px', 'height': '650px', 'display': 'inline-block',
                       'vertical-align': 'top', 'border': '1px solid black',
                       'padding': '20px'}),
            html.Div(children=[
                dcc.Graph(id='my_graph'),
            ],
                style={'width': '700px', 'height': '650px', 'display': 'inline-block'}),
        ]), ],
    style={'text-align': 'center', 'display': 'inline-block', 'width': '100%'})


@app.callback(
    Output(component_id='my_graph', component_property='figure'),
    Input(component_id='user_song', component_property='value'))
def update_plot(input_song):
    if input_song:
        doc = df_train.iloc[input_song].values.reshape(1, -1)
        doc_1 = df_train.iloc[input_song].values.reshape(1, -1)
    print(doc.flags)
    print(doc_1.flags)
    print(doc_1.shape)

    # Query Using K-Nearest Neighbors
    __, neigh_index = knn_loader.kneighbors(doc)

    df_viz = df_train.iloc[neigh_index[0][:4]]
    # Drop age columns as unnecessary for visualization
    df_viz = df_viz[df_viz.columns.drop(list(df_viz.filter(regex='age')))]
    df_viz_transposed = df_viz.transpose().reset_index()

    df_t1 = pd.DataFrame(dict(r=df_viz_transposed.iloc[:, 1],
                              theta=df_viz_transposed.iloc[:, 0]))
    fig = px.line_polar(df_t1, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
