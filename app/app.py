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
#colors = itertools.cycle(['#6c503e', '#446652','#8C93A8', '#EDCB96', '#B5C2B7'])
colors = ['#6c503e', '#446652','#8C93A8', '#EDCB96', '#B5C2B7']


###### MAIN TAB BLOCKS ######

predict_text = dcc.Markdown('''
### About:
This dashboard is a demonstration of the
[Bird Vocalization Classification](https://github.com/benbogart/bird_vocalization_classification)
project by Ben Bogart.  You view the full code in this
[github repository](https://github.com/benbogart/bird_vocalization_classification) where there
is a detailed description of the project.

There are two tabs in this dashboard, `Prediction Demo` and `Explore Models`.
- This is the `Prediction Demo` tab.  Below you can select one of the provided audio
samples or upload your own to test the prediction and see the results of the model.

- On the `Explore Models` tab you can dig into the various models created in this project
and explore the training, validation, and test metrics for each.
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
                     [xeno-canto.org](https://www.xeno-canto.org/11464)'''),
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


explore_block = html.Div([
    dcc.Graph(id='test_accuracy'),
    dcc.Dropdown(id='model_selector',
                 options=[{'label':v, 'value':v} for v in df_metrics['model_name'].unique()],
                 value=['milsed_7block_dense'],
                 multi=True
    ),
    dcc.Dropdown(id='data_selector',
                 options=[{'label':v, 'value':v} for v in df_metrics['data_subset'].unique()],
                 value=['kaggle_full_length_npy_aug'],
                 multi=True
    ),
    html.Div(id='explore_header'),
    html.Div(id='accuracy_plot'),
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

    model = MODELS['milsed_7block_dense']()
    weight_file = 'milsed_7block_dense-birdsongs_2_1618597659_27bfc8cd.h5'


    # stringlist = []
    # model.summary(print_fn=lambda x: stringlist.append(x))
    # model_summary = "\n".join(stringlist)

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

    graph_block = get_wavplot(signal)
    model.load_weights(os.path.join('data', weight_file))
    pred = model.predict(signal.reshape(1,-1,1))
    print(pred.shape)
    print(type(pred))
    print(pred.argmax())
    y_pred = pred.argmax()

    images = get_photos(translation['sci_name'][y_pred])
    images_block = html.Div([
        html.H3('Images from flikr'),
        html.Div(get_photos(translation['sci_name'][y_pred]))
        ], className='images'
    )

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
            images_block,
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
        'tags':name,
        'extras':'owner_name, url_s',
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

    waveform = px.line(signal, color_discrete_sequence=[colors[1]])
    waveform.update_yaxes(title={'text':'magnitude'})
    waveform.update_xaxes(title={'text':'samples'})
    waveform.update_layout(transparent_layout,
                           title='Waveform',
                           showlegend=False,
                           font = {'color':colors[0]})
    # waveform.update_layout(title='Waveform')
    # waveform.update_layout(showlegend=False)

    # stft = librosa.stft(signal, n_fft=2048,  hop_length=512)
    # spectrogram = np.abs(stft)
    # mel_spectrogram = librosa.amplitude_to_db(spectrogram)

    S = librosa.feature.melspectrogram(signal, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    mel_spectrogram = librosa.power_to_db(S, ref=np.max)
    mspec = px.imshow(mel_spectrogram, aspect='auto', origin='lower')
    mspec.update_yaxes(title={'text':'mels'})
    mspec.update_xaxes(title={'text':'windows'})
    mspec.update_layout(transparent_layout,
                        title='Mel Spectrogram',
                        font = {'color':colors[0]})
    # mspec.update_layout(title='Mel Spectrogram')

    # mspec.update_yaxes(autorange="reversed")
    graph_block = html.Div([
        dcc.Graph(id='waveform', figure=waveform, className='col nopad'),
        dcc.Graph(id='mspec', figure=mspec, className='col nopad')
    ], className='row pred-plots')

    return graph_block

###### Explore Model Functions ######

@app.callback(Output('test_accuracy', 'figure'),
              Input('test_accuracy', 'id'))
def get_test_accuracy_plot(values):

    # combine model name and Data Subset
    df_plot = df_metrics.copy()
    df_plot['model'] = df_plot['model_name'] + '<br>' + df_plot['data_subset']
    df_plot['model_num'] = range(1, len(df_plot) + 1)
    df_plot['model_num_text'] = 'Model #' + df_plot['model_num'].astype('str')

    # limit DataFrame to only columns we need for plot
    df_plot = df_plot[['model_name',
                   'data_subset',
                   'model_num_text',
                   'test_categorical_accuracy',
                   'test_top_3_accuracy',
                   'test_top_5_accuracy']]

    # rename columns for plot
    df_plot.columns = ['Model Name',
                       'Data',
                       'Model No.',
                       'Categorical Accuracy',
                       'Top 3 Categorical Accuracy',
                       'Top 5 Categorical Accuracy']

    # flatten dataframe for plot
    df_plot = df_plot.melt(id_vars=['Model Name', 'Data', 'Model No.'],
                           var_name='Accuracy Metric',
                           value_vars=['Categorical Accuracy',
                                          'Top 3 Categorical Accuracy',
                                          'Top 5 Categorical Accuracy'],
                           value_name='Accuracy')

    # accuracy plot with plotly for dashboard
    fig = px.bar(df_plot.sort_values('Accuracy'), x='Model No.',
                 y='Accuracy', color='Accuracy Metric',
                 barmode='group',
                 hover_name='Model No.',
                 hover_data=['Model Name', 'Data'],
                 color_discrete_sequence=colors,
                 title='Test Accuracy by model name')

    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=1.05,
            xanchor="left",
            x=0.01
        ),
        annotations = [
            dict(xref='paper',
                 yref='paper',
                 x=0, y=1.13,
                 showarrow=False,
                 text ='Hover to see Model Name and Dataset'
            )
        ],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font = {'color':'black'}
    )

    return fig


@app.callback(Output('accuracy_plot', 'children'),
              Input('model_selector', 'value'),
              Input('data_selector', 'value'))
def get_accuracy_plot(models, data_subsets):
    # value contains model names

    if len(models) == 0 or len(data_subsets) == 0:
        return html.Div('Please select a model above', className='notice')

    acc_fig = go.Figure()
    loss_fig = go.Figure()
    iter_colors = itertools.cycle(['#6c503e', '#446652','#8C93A8',
                                   '#EDCB96', '#B5C2B7'])

    # maintain count of number of models displayed
    model_count = 0

    for model_name in models:
        for data in data_subsets:

            model_stats = df_metrics[(df_metrics['model_name'] == model_name) &
                                     (df_metrics['data_subset'] == data)]

            if len(model_stats) > 0:
                model_count += 1

                df_acc = model_stats[[
                    'model_name',
                    'categorical_accuracy',
                    'data_subset'
                ]].explode(
                    'categorical_accuracy'
                ).reset_index()
                #df_acc.index = pd.Series(range(1, len(df_acc) + 1), name='Epochs')
                #df_acc['model'] = best_model['model_name'] + '/' + best_model['data_subset']

                df_acc_val = model_stats[[
                    'model_name', 'val_categorical_accuracy'
                ]].explode('val_categorical_accuracy').reset_index()



                df_loss = model_stats[['model_name',
                     'loss']].explode('loss').reset_index()

                df_loss_val = model_stats[['model_name',
                     'val_loss']].explode('val_loss').reset_index()


                df_plt = pd.concat([df_acc,
                                    df_acc_val['val_categorical_accuracy'],
                                    df_loss['loss'],
                                    df_loss_val['val_loss']], axis=1)
                df_plt['Epochs'] = range(1, len(df_plt)+1)
                df_plt['model_name'] = df_plt['model_name'] + '<br>' + df_plt['data_subset']


                color = next(iter_colors)

                acc_fig.add_trace(go.Scatter(y=df_plt['categorical_accuracy'],
                                 x=df_plt['Epochs'],
                                 name='<b>Accuracy:</b><br>' + df_plt['model_name'].iloc[0],
                                 line=dict(color=color, dash='solid')))

                acc_fig.add_trace(go.Scatter(y=df_plt['val_categorical_accuracy'],
                                 x=df_plt['Epochs'],
                                 name='<b>Val. Accccuracy:</b><br>' + df_plt['model_name'].iloc[0],
                                 line=dict(color=color, dash='dash')))

                acc_fig.update_yaxes(title={'text':'Accuracy'},
                                     gridcolor='rgba(0,0,0,0.2)')
                acc_fig.update_xaxes(title={'text':'Epochs'},
                                     gridcolor='rgba(0,0,0,0.2)')
                acc_fig.update_layout(go.Layout(title="Accuracy vs Validation Accuracy"),
                                      font = {'color':'black'},
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')

                loss_fig.add_trace(go.Scatter(y=df_plt['loss'],
                                 x=df_plt['Epochs'],
                                 name='<b>Loss:</b><br>' + df_plt['model_name'].iloc[0],
                                 line=dict(color=color, dash='solid')))

                loss_fig.add_trace(go.Scatter(y=df_plt['val_loss'],
                                 x=df_plt['Epochs'],
                                 name='<b>Val. Loss:</b><br>' + df_plt['model_name'].iloc[0],
                                 line=dict(color=color, dash='dash')))

                loss_fig.update_yaxes(title={'text':'Loss'},
                                      gridcolor='rgba(0,0,0,0.2)')
                loss_fig.update_xaxes(title={'text':'Epochs'},
                                      gridcolor='rgba(0,0,0,0.2)')
                loss_fig.update_layout(go.Layout(title="Loss vs Validation Loss"),
                                       font = {'color':'black'},
                                       paper_bgcolor='rgba(0,0,0,0)',
                                       plot_bgcolor='rgba(0,0,0,0)')


    if model_count > 0:
        children = [dcc.Graph(id='train_accuracy', figure=acc_fig),
                    dcc.Graph(id='val_accuracy', figure=loss_fig)]
    else:
        children = [html.Div('No models match your search', className='notice')]

    return children



def get_accuracy_plot_bak(value):

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




if __name__ == "__main__":

    # gunicorn.run(app, port=8050, host='0.0.0.0')
    # app.run_server(debug=True, port=8000, host='0.0.0.0')

    app.run_server(debug=True,
                   host="0.0.0.0",
                   port=port)
