# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ["app.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

predict_block = html.Div(
    [
        html.Div('Hello World')
    ],
    id='predict-block')




app.layout = html.Div(
    [
        predict_block
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
