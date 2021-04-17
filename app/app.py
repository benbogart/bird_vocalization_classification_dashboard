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

#import tensorflow as tf
#from models import MODELS

import gunicorn

app = dash.Dash(__name__)
server = app.server
#server = flask.Flask(__name__) # define flask app.server

# model = MODELS['milsed_7block_dense']()
#
# stringlist = []
# model.summary(print_fn=lambda x: stringlist.append(x))
# model_summary = "\n".join(stringlist)
#
# with open('resources/data_kag_split_single_label.json', 'r') as f:
#     labels = json.load(f)
#
# label_map = labels['mapping']
#
# predict_block = html.Div([
#     html.Div([
#         html.Ol([
#             html.Li('File 1'),
#             html.Li('File 2'),
#             html.Li(
#                 html.Audio([
#                     html.Source(src='https://www.xeno-canto.org/sounds/uploaded/VJTMVNCHXA/XC528037-NWS02_ColumbiaNWR_0610-2020_AmericanBittern_XC.mp3',
#                                 type='audio/mpeg'),
#                     'Your browser does not support the audio element'
#                 ], controls=True)
#             ),
#             html.Li([
#                 html.Iframe(src='https://www.xeno-canto.org/419247/embed',
#                     width='340', height='220', className='audio-embed'
#                 ),
#                 html.Button('Select', id='button1', n_clicks=0, className='button')
#             ]),
#             html.Li(
#                 dcc.Upload(
#                     id='upload-data',
#                     children=html.Div([
#                         'Upload your own audio file.  Predicion will be made on the last 10 seconds of the file',
#                         html.Br(),
#                         'Drag and Drop or ',
#                         html.A('Select Files')
#                     ]),
#                     multiple=False
#                 )
#             )
#         ])
#     ], className='left'),
#
#     dcc.Loading(
#         html.Div(id='output', children=[
#         'This is the amazing origional content of the right side of the page!',
#         html.Pre(model_summary)]),
#         fullscreen=True,
#         style={'opacity': 0.4}
#     )
# ], id='predict-block2')
# #
app.layout = html.Div(
    [
        'hola'
        # predict_block
    ]
)

# @app.callback(Output('output', 'children'),
#               Input('button1', 'n_clicks'),
#               Input('upload-data', 'contents'),
#               Input('upload-data', 'filename'))
# def predict(btn1, contents, filename):
#
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         raise PreventUpdate
#
#     trigger = ctx.triggered[0]['prop_id'].split('.')[0]
#
#     if trigger == 'button1':
#         with open('data/sometext.txt', 'r') as f:
#             text = f.read()
#         return 'You select buton 1!<br>' + text
#
#     elif trigger == 'upload-data':
#         signal = parse_contents(contents, filename)
#
#         fig = px.line(signal)
#         wav_graph = dcc.Graph(id='signal', figure=fig)
#
#         pred = model.predict(signal.reshape(1,-1,1))
#         print(pred.shape)
#         print(type(pred))
#         print(pred.argmax())
#         return [wav_graph, label_map[pred.argmax()]]
#
# def parse_contents(contents, filename):
#     content_type, content_string = contents.split(',')
#
#     decoded = base64.b64decode(content_string)
#     with open(filename, "wb") as f:
#         f.write(decoded)
#
#     signal, sr = librosa.load(filename)
#
#     os.remove(filename)
#     return signal[-220500:]

if __name__ == "__main__":

    # gunicorn.run(app, port=8050, host='0.0.0.0')
    app.run_server(debug=True, port=8000, host='0.0.0.0')
