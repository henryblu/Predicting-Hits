import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

FEATURE_COLUMNS = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'mode', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
SCALER_POP = joblib.load(filename = './objects/scaler_pop')
SCALER_FLOATS = joblib.load(filename = './objects/scaler_floats')
MODEL = joblib.load(filename = './objects/predictor')
PLAYLIST_NAME_TO_URL = joblib.load(filename = './objects/playlist_name_to_url')

'''
Given the url, returns unique id of the song
'''
def url_to_id(url):
    return url.split("/")[-1].split("?")[0]

'''
Params: spotify client and the song ID
Returns: Normalized data of the song
'''
def process_data(sp, id):

    song = sp.track(id)
    popularity = song['popularity']
    audio_features = sp.audio_features(id)

    data = [[popularity] + list(audio_features[0].values())]
    index = ['popularity'] + list(audio_features[0].keys())
    data = pd.DataFrame(data = data, columns = index)
    data = data[FEATURE_COLUMNS]

    def onehot_prep(df, column, new_name):
        tf_df = pd.get_dummies(df[column])
        feature_names = tf_df.columns
        tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
        tf_df.reset_index(drop = True, inplace = True)
        return tf_df


    key_onehot = onehot_prep(data, 'key', 'key') * 0.5
    key_columns = ['key|0', 'key|1', 'key|2', 'key|3', 'key|4', 'key|5', 'key|6', 'key|7', 'key|8', 'key|9', 'key|10', 'key|11']

    for i in key_columns:
        if i not in key_onehot:
            key_onehot[i] = 0

    mode_onehot = onehot_prep(data, 'mode', 'mode') * 0.5
    mode_columns = ['mode|0', 'mode|1']
    for i in mode_columns:
        if i not in mode_onehot:
            mode_onehot[i] = 0

    pop = data[['popularity']]
    pop_scaled = pd.DataFrame(SCALER_POP.transform(np.array(pop).reshape(-1,1)) * 0.2, columns = pop.columns)

    floats = data[['danceability', 'energy', 'loudness', 'speechiness','acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
    floats_scaled = pd.DataFrame(SCALER_FLOATS.transform(np.array(floats).reshape(1,-1)) * 0.2, columns = floats.columns)
    
    processed = pd.concat([floats_scaled, pop_scaled, key_onehot, mode_onehot], axis = 1)

    return processed

'''
Given the processed data, it feeds the song to the model which predicts the top 5 closest playlists that sound the same
params: Processed DataFrame row
return: Dictionary with key value pair as {playlist name: url}
'''
def predict_playlists(processed):
    #an array of the audio features
    clf = MODEL
    
    n = 5

    probas = clf.predict_proba(processed)
    print('Found similar playlists!')
    top_n_lables_idx = np.argsort(-probas, axis=1)[:, :n]
    top_n_labels = [clf.classes_[i] for i in top_n_lables_idx]

    name_to_url = {name: PLAYLIST_NAME_TO_URL[name] for name in top_n_labels[0]}
    
    return name_to_url

'''
The test function can be used for trying out the functionality 
'''
def main(sp):

    url = input('Give song url: ')
    id = url_to_id(url)
    print(id)

    print('Starting to fetch the audio features')
    processed_data = process_data(sp, id)

    print('Data processed, fitting to the model...')
    playlists = predict_playlists(processed_data)

    print(playlists)


if __name__ == "__main__":
    #Make a config file to hide the client secret
    SPOTIPY_CLIENT_ID = ''
    SPOTIPY_CLIENT_SECRET = ''
    REDIRECT_URI = 'http://localhost:7000/callback'
    scope = "user-library-read"

    cache_handler = spotipy.cache_handler.MemoryCacheHandler()
    auth_manager = SpotifyClientCredentials(client_id = SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET, cache_handler=cache_handler)
    sp = spotipy.Spotify(auth_manager = auth_manager)
    main(sp)