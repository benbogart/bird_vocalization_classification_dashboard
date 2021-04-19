# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import librosa
import json

import base64
import io
import os
import numpy as np

import tensorflow as tf
from models import MODELS

import gunicorn

import requests

app = dash.Dash(__name__)
port = int(os.environ.get("PORT", 5000))
# server = app.server


model = MODELS['cnn1_audin_drp_1']()
weight_file = 'cnn1_audin_drp_1-birdsongs_2_1618153718_0f7f4e95.h5'

stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
model_summary = "\n".join(stringlist)

with open('resources/data_kag_split_single_label.json', 'r') as f:
    labels = json.load(f)

label_map = labels['mapping']

predict_text = dcc.Markdown('''
### About:
This dashboard is a demonstration of the birdsong_detection by Ben Bogart.  You
view the full code in this
[github repository](https://github.com/benbogart/birdsong_detection) where there
is a detailed description of the project.

There are two tabs in this dashboard, predictions and results.
- This is the `Prediction Demo` tab.  Below you can select one of the provided audio
samples or upload your own to test the prediction and see the results of the model.

- On the `Explore Results` tab you can dig into the various models created in this project
and explore the training, validation, and test statistics for each.
''')

# XC408484
# XC124819
# XC135462
# XC441497
#

audio_samples = html.Ul([
    dcc.Markdown('''## or choose a recording below'''),
    html.Li([
        dcc.Markdown('''Ash-throated Flycatcher from
                        [xeno-canto.org](https://www.xeno-canto.org/441497)'''),
        html.Audio([
            html.Source(src='https://www.xeno-canto.org/sounds/uploaded/KZYUWIRZVH/XC441497-FLYCATC_Ash-thr_c%20Mesa%20del%20Campanero%202100m%20051618%201014.mp3',
                        type='audio/mpeg'),
            'Your browser does not support the audio element'
        ], controls=True),
        html.Br(),
        html.Button('Predict Bird from this Recording',
                    n_clicks=0, id='button2', className='button'),
    ]),
    html.Li([
        dcc.Markdown('''American Redstart from
                        [xeno-canto.org](https://www.xeno-canto.org/135462)'''),
        html.Audio([
            html.Source(src='https://www.xeno-canto.org/sounds/uploaded/PWDLINYMKL/XC135462-American%20Redstart1301.mp3',
                        type='audio/mpeg'),
            'Your browser does not support the audio element'
        ], controls=True),
#        html.Br(),
        html.Button('Predict Bird from this Recording',
                    n_clicks=0, id='button3', className='button'),
    ]),
    html.Li([
        dcc.Markdown('''Barn Swallow from
                     [xeno-canto.org](https://www.xeno-canto.org/11464)
(This file will predict the wrong bird type)'''),
        html.Audio([
            html.Source(src='https://www.xeno-canto.org/sounds/uploaded/CLKPHLYUHA/TMIItapeIVHirRus.mp3'),
            'Your browser does not support the audio element'
        ], controls=True),
        html.Button('Predict Bird from this Recording',
                    n_clicks=0, id='button1', className='button'),
    ]),
], id='audio_samples')

upload_sample = html.Div([
    dcc.Markdown('''## Upload your own audio file...
Predicion will be made on the last 10 seconds of your file'''),
    html.Br(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        multiple=False
    )
])

predict_block = dcc.Loading(
    html.Div([
        predict_text,

        html.Div([
            html.Div([
                upload_sample
            ], className="col"),
            html.Div([
                audio_samples
            ], className="col"),
        ], className='row')
    ], id='main-predict'),
type='cube', className='loading'
)

app.layout = html.Div(
    [
        html.Div([
            html.H1('Birdsong Detection'),
            html.P('by Ben Bogart'),
        ],  id='title'),
        dcc.Tabs([
            dcc.Tab(label='Prediction Demo',
                    children=predict_block),
            dcc.Tab(label='Explore Models',
                    children='This is the explore models tab')
        ], className='tabs')
    ]
)

@app.callback(Output('main-predict', 'children'),
              Input('button1', 'n_clicks'),
              Input('button2', 'n_clicks'),
              Input('button3', 'n_clicks'),
              Input('upload-data', 'contents'),
              Input('upload-data', 'filename'))
def predict(btn1, btn2, btn3, contents, filename):

    button_files = {'button1':'XC11464.mp3',
                    'button2':'XC441497.mp3',
                    'button3':'XC135462.mp3'}

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger in ['button1', 'button2', 'button3']:
        filename = button_files[trigger]
        signal, sr =librosa.load(os.path.join('data', filename))
    elif trigger == 'upload-data':
        signal = parse_contents(contents, filename)

    # get last 10 seconds of file
    signal = signal[-220500:]

    # load label translation json
    with open('resources/translation.json', 'r') as f:
        translation = json.load(f)

    # TODO create figure with both melspec and signal
    # fig = px.line(signal)
    # wav_graph = dcc.Graph(id='signal', figure=fig, className='signal')

    # wav_plot = dcc.Graph(id='signal', figure=get_wavplot(signal), className='signal')
    # get_wavplot(signal)

    graph_block = get_wavplot(signal)
    model.load_weights(os.path.join('data', weight_file))
    pred = model.predict(signal.reshape(1,-1,1))
    print(pred.shape)
    print(type(pred))
    print(pred.argmax())
    y_pred = pred.argmax()

    images = get_photos(translation['sci_name'][y_pred])

    header_block = html.Ul([
        html.H1(f"Bird species: {translation['species'][y_pred]}"),
        html.Li(f"Scientific Name: {translation['sci_name'][y_pred]}"),
        html.Li([
            "eBird Code: ",
            html.A(
                f"{translation['ebird_code'][y_pred]}",
                href=f"https://ebird.org/species/{translation['ebird_code'][y_pred]}"
            )
        ]),
        html.Li([
            'Wikipedia: ',
            html.A(
                f"{translation['species'][y_pred]}",
                href=f"https://en.wikipedia.org/wiki/{translation['species'][y_pred].replace(' ', '_')}"
            )
        ]),
        html.Li(f'Filename: {filename}')
    ])

    try_again_block = html.Div(
        html.A(
            html.Button('Try Again!'),
            href=''
        ),
        className='try-again'
    )

    return [
            header_block,
            graph_block,
            html.Div(images),
            try_again_block
            #dcc.Markdown('[Try Again!]()', className='button')
            ]

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    with open(filename, "wb") as f:
        f.write(decoded)

    signal, sr = librosa.load(filename)

    os.remove(filename)
    return signal

def get_photos(name):


    print('Getting photos for ', name)
    flikr_key = '9ed6feedf4ed3df4f12101958364b9bb'

    query = {
        'method':'flickr.photos.search',
        'format':'json',
        'api_key':flikr_key,
        'tags':'Setophaga ruticilla',
        'extras':'description, license, date_upload, date_taken, owner_name, icon_server, original_format, last_update, geo, tags, machine_tags, o_dims, views, media, path_alias, url_sq, url_t, url_s, url_q, url_m, url_n, url_z, url_c, url_l, url_o',
        'per_page':8,
        "nojsoncallback":"1",
        'sort':'relevance'
    }

    r = requests.get('https://www.flickr.com/services/rest/', params=query)

    if not r.ok:
        return dcc.Markdown('**Could not load Flickr images, please try again later**')

    r = r.json()
    images = r['photos']['photo']

    print(f'There are {len(images)} photos to display')
    img_blocks = []
    for img in images:
        img_blocks.append(
            html.Div([
                html.A(
                    html.Img(src=img['url_s']),
                    href=f"https://flickr.com/photos/{img['owner']}/{img['id']}"
                ),
                html.P(
                    'Image © ' + img['ownername'],
                    className='caption'
                )

            ], className='image')
        )


    return img_blocks

def get_wavplot(signal):

    transparent_layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    waveform = px.line(signal)
    waveform.update_yaxes(title={'text':'magnitude'})
    waveform.update_xaxes(title={'text':'samples'})
    waveform.update_layout(transparent_layout)
    waveform.update_layout(title='Waveform')

    # stft = librosa.stft(signal, n_fft=2048,  hop_length=512)
    # spectrogram = np.abs(stft)
    # mel_spectrogram = librosa.amplitude_to_db(spectrogram)

    S = librosa.feature.melspectrogram(signal, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    mel_spectrogram = librosa.power_to_db(S, ref=np.max)
    mspec = px.imshow(mel_spectrogram, aspect='auto', origin='lower')
    mspec.update_yaxes(title={'text':'mels'})
    mspec.update_xaxes(title={'text':'windows'})
    mspec.update_layout(transparent_layout)
    mspec.update_layout(title='Mel Spectrogram')

    # mspec.update_yaxes(autorange="reversed")
    graph_block = html.Div([
        dcc.Graph(id='waveform', figure=waveform, className='col nopad'),
        dcc.Graph(id='mspec', figure=mspec, className='col nopad')
    ], className='row pred-plots')

    return graph_block

# def get_wavplot(signal):
#     fig = make_subplots(rows=2, cols=1)
#
#     fig.add_trace(
#         Go.Scatter(y=signal),
#         row=1, col=1
#     )
#
#     stft = librosa.stft(signal, n_fft=2048,  hop_length=512)
#     spectrogram = np.abs(stft)
#     mel_spectrogram = librosa.amplitude_to_db(spectrogram)
#
#     fig.add_trace(
#         go.Heatmap(z=mel_spectrogram),
#         row=2, col=1
#     )
#
#     return fig
if __name__ == "__main__":

    # gunicorn.run(app, port=8050, host='0.0.0.0')
    # app.run_server(debug=True, port=8000, host='0.0.0.0')

    app.run_server(debug=True,
                   host="0.0.0.0",
                   port=port)
