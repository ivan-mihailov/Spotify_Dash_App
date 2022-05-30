# spotify-song-rec
Plotly Dash Recommendation App (static) for Spotify Songs with Radar Vizualizations

The Dash App works locally (use app_local.py) and renders beautifully.

When deployed to Heroku, however, the app crashes because it uses too much memory to store the dataframes used for the machine learning model and the radar visualizations. To properly deploy the app in its current state requires the Enterprise tier of Heroku which is outside of the budget for this project.

TODO:
Explore ways to reduce the memory load and allow deployment on Heroku Free tier level.
