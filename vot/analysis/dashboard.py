
from typing import List

import dash
import dash_core_components as dcc
import dash_html_components as html

from vot.workspace import Workspace
from vot.tracker import Tracker
from vot.dataset import Sequence

def run_dashboard(wokrspace: Workspace, trackers: List[Tracker], debug=False):
    app = dash.Dash(__name__)

    app.title = "VOT Analysis Dasboard"
    app.layout = html.Div(children=[
        html.H1(children='VOT Analysis Dasboard'),

        html.Div(children='''
            Workspace path: {}
        '''.format(Workspace.directory)),

        dcc.Tabs([dcc.Tab()]),


        dcc.Graph(
            id='example-graph',
            figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                ],
                'layout': {
                    'title': 'Dash Data Visualization'
                }
            }
        )
    ])


    app.run_server(debug=debug)
 