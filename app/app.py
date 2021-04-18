# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import plotly.express as px
import librosa
import json

import base64
import io
import os

import tensorflow as tf
from models import MODELS

import gunicorn

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
# Birdsong Detection

This dashboard is a demonstration of the birdsong_detection by Ben Bogart.  You
view the full code in this
[github repository](https://github.com/benbogart/birdsong_detection) where there
is a detailed readme file describing how the project was accomplished.

There are two tabs in this dashboard, predictions and results.
- This is the `predictions` tab.  Below you can select one of the provided audio
samples or upload your own to test the prediction and see the results of the model.

- On the `results` tab you can dig into the various models created in this project
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
className='loading'
)


# output_block = dcc.Loading(
#         html.Div(id='output', children=[
#         'This is the most spectactularly arrogant origional content of the right side of the page!',
#         html.Pre(model_summary)]),
#         fullscreen=True,
#         style={'opacity': 0.4}
#     )

app.layout = html.Div(
    [
        # 'hola'
        predict_block
    ]
)

@app.callback(Output('main-predict', 'children'),
              Input('button1', 'n_clicks'),
              Input('button2', 'n_clicks'),
              Input('button3', 'n_clicks'),
              Input('upload-data', 'contents'),
              Input('upload-data', 'filename'))
def predict(btn1, btn2, btn3, contents, filename):

    button_files = {'button1':'data/XC11464.mp3',
                    'button2':'data/XC441497.mp3',
                    'button3':'data/XC135462.mp3'}

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger in ['button1', 'button2', 'button3']:
        filename = button_files[trigger]
        signal, sr =librosa.load(filename)
    elif trigger == 'upload-data':
        signal = parse_contents(contents, filename)

    # get last 10 seconds of file
    signal = signal[-220500:]


    # TODO create figure with both melspec and signal
    fig = px.line(signal)
    wav_graph = dcc.Graph(id='signal', figure=fig, className='signal')
    model.load_weights(os.path.join('data', weight_file))
    pred = model.predict(signal.reshape(1,-1,1))
    print(pred.shape)
    print(type(pred))
    print(pred.argmax())
    return [
            filename,
            wav_graph,
            label_map[pred.argmax()],
            dcc.Markdown('[Try Again!]()')
            ]

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    with open(filename, "wb") as f:
        f.write(decoded)

    signal, sr = librosa.load(filename)

    os.remove(filename)
    return signal

if __name__ == "__main__":

    # gunicorn.run(app, port=8050, host='0.0.0.0')
    # app.run_server(debug=True, port=8000, host='0.0.0.0')

    app.run_server(debug=True,
                   host="0.0.0.0",
                   port=port)
