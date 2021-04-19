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
import pandas as pd
import itertools

import gunicorn

import requests

app = dash.Dash(__name__)
port = int(os.environ.get("PORT", 5000))
# server = app.server

# create color loop for plots
colors = itertools.cycle(['red', 'green', 'blue'])


###### MAIN TAB BLOCKS ######

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

###### Explore Models Tab ######

explore_header = 'This is the explore models tab 2'

# accuracy_plot = get_accuracy_plot()
df_metrics = pd.read_pickle('data/df_metrics.pkl')

# def get_test_accuracy_plot():
#
#     # value contains model names
#     fig = go.Figure()
#     for model_name in value:
#         # get the last row for a model in case there are more than one
#         model_stats = df_metrics[df_metrics['model_name'] == model_name].iloc[-1:]
#         df_acc = model_stats[['model_name',
#              'categorical_accuracy']].explode('categorical_accuracy').reset_index()
#
#         df_val = model_stats[['model_name',
#              'val_categorical_accuracy']].explode('val_categorical_accuracy').reset_index()
#
#         df_plt = pd.concat([df_acc, df_val['val_categorical_accuracy']], axis=1)
#         df_plt['Epochs'] = range(1, len(df_plt)+1)
#
#         color = next(colors)
#
#         fig.add_trace(go.Scatter(y=df_plt['categorical_accuracy'],
#                          x=df_plt['Epochs'],
#                          name='accuracy: ' + df_plt['model_name'].iloc[0],
#                          line=dict(color=color, dash='solid')))
#
#         fig.add_trace(go.Scatter(y=df_plt['val_categorical_accuracy'],
#                          x=df_plt['Epochs'],
#                          name='val accuracy: ' + df_plt['model_name'].iloc[0],
#                          line=dict(color=color, dash='dash')))
#
#         fig.update_yaxes(title={'text':'Accuracy'})
#         fig.update_xaxes(title={'text':'Epochs'})
#         fig.update_layout(go.Layout(title="Accuracy vs Validation Accuracy"))
#
#
#     g = dcc.Graph(id='the-graph', figure=fig)
#     return g



explore_block = html.Div([
    # dcc.Dropdown(id='test_accuracy_selector',
    #              options=[{'label':'Cagegorical Accuracy',
    #                        'value':'test_categical_acuracy'},
    #                       {'label':'Top 3 Cagegorical Accuracy',
    #                        'value':'test_top_3_accuracy'},
    #                       {'label':'Top 5 Cagegorical Accuracy',
    #                        'value':'test_top_5_categical_acuracy'}],
    #              value=['test_categical_acuracy'],
    #              multi=True
    # ),
    dcc.Graph(id='test_accuracy'),
    dcc.Dropdown(id='model_selector',
                 options=[{'label':v, 'value':v} for v in df_metrics['model_name'].unique()],
                 value=['milsed_7block_dense'],
                 multi=True
    ),
    html.Div(id='explore_header'),
    dcc.Graph(id='train_accuracy'),
    dcc.Graph(id='val_accuracy'),
    html.Div(id='accuracy_model_graph')
])

###### Page Layout ######

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
                    children=explore_block)
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
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # importing here improves page load time.
    import tensorflow as tf
    from models import MODELS

    model = MODELS['cnn1_audin_drp_1']()
    weight_file = 'cnn1_audin_drp_1-birdsongs_2_1618153718_0f7f4e95.h5'

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)

    with open('resources/data_kag_split_single_label.json', 'r') as f:
        labels = json.load(f)

    label_map = labels['mapping']

    button_files = {'button1':'XC11464.mp3',
                    'button2':'XC441497.mp3',
                    'button3':'XC135462.mp3'}

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

###### Explore Model Functions ######

@app.callback(Output('train_accuracy', 'figure'),
              Output('val_accuracy', 'figure'),
              Input('model_selector', 'value'))
def get_accuracy_plot(value):

    # value contains model names

    colors = itertools.cycle(['red', 'green', 'blue'])

    acc_fig = go.Figure()
    val_fig = go.Figure()
    for model_name in value:
        # get the last row for a model in case there are more than one
        model_stats = df_metrics[df_metrics['model_name'] == model_name].iloc[-1:]
        df_acc = model_stats[['model_name',
             'categorical_accuracy']].explode('categorical_accuracy').reset_index()

        df_acc_val = model_stats[['model_name',
             'val_categorical_accuracy']].explode('val_categorical_accuracy').reset_index()

        df_loss = model_stats[['model_name',
             'loss']].explode('loss').reset_index()

        df_loss_val = model_stats[['model_name',
             'val_loss']].explode('val_loss').reset_index()


        df_plt = pd.concat([df_acc,
                            df_acc_val['val_categorical_accuracy'],
                            df_loss['loss'],
                            df_loss_val['val_loss']], axis=1)
        df_plt['Epochs'] = range(1, len(df_plt)+1)

        color = next(colors)

        acc_fig.add_trace(go.Scatter(y=df_plt['categorical_accuracy'],
                         x=df_plt['Epochs'],
                         name='accuracy: ' + df_plt['model_name'].iloc[0],
                         line=dict(color=color, dash='solid')))

        acc_fig.add_trace(go.Scatter(y=df_plt['val_categorical_accuracy'],
                         x=df_plt['Epochs'],
                         name='val accuracy: ' + df_plt['model_name'].iloc[0],
                         line=dict(color=color, dash='dash')))

        acc_fig.update_yaxes(title={'text':'Accuracy'})
        acc_fig.update_xaxes(title={'text':'Epochs'})
        acc_fig.update_layout(go.Layout(title="Accuracy vs Validation Accuracy"))

        val_fig.add_trace(go.Scatter(y=df_plt['loss'],
                         x=df_plt['Epochs'],
                         name='loss: ' + df_plt['model_name'].iloc[0],
                         line=dict(color=color, dash='solid')))

        val_fig.add_trace(go.Scatter(y=df_plt['val_loss'],
                         x=df_plt['Epochs'],
                         name='val loss: ' + df_plt['model_name'].iloc[0],
                         line=dict(color=color, dash='dash')))

        val_fig.update_yaxes(title={'text':'Loss'})
        val_fig.update_xaxes(title={'text':'Epochs'})
        val_fig.update_layout(go.Layout(title="Loss vs Validation Loss"))

    return acc_fig, val_fig

@app.callback(Output('test_accuracy', 'figure'),
              Input('test_accuracy', 'id'))
def get_test_accuracy_plot(values):

    df_plot = df_metrics[['model_name',
                         'test_categorical_accuracy',
                         'test_top_3_accuracy',
                         'test_top_5_accuracy']]

    df_plot.columns = ['Model Name',
                                 'Categorical Accuracy',
                                 'Top 3 Categorical Accuracy',
                                 'Top 5 Categorical Accuracy']



    df_plot = df_plot.melt(id_vars=['Model Name'],
                           var_name='Accuracy Metric',
                           value_vars=['Categorical Accuracy',
                                          'Top 3 Categorical Accuracy',
                                          'Top 5 Categorical Accuracy'],
                           value_name='Accuracy')


    fig = px.bar(df_plot, x='Model Name',
                 y='Accuracy', color='Accuracy Metric',
                 barmode='group',
                 title='Test Accuracy by model name')

    return fig


if __name__ == "__main__":

    # gunicorn.run(app, port=8050, host='0.0.0.0')
    # app.run_server(debug=True, port=8000, host='0.0.0.0')

    app.run_server(debug=True,
                   host="0.0.0.0",
                   port=port)
