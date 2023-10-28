
import torch 
torch.set_num_threads(1)

import dash 
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from digit_classifier_NN import (
    MLP,
    MLP_model_fp,
    MLP_model_fp_2,
    MLP_model_fp_3,
    train_images, 
    test_images, 
    train_labels, 
    test_labels,
    train_NN_process
)
from plot_NN import *

html_fp = "NN-app.html"

state_dict = torch.load(MLP_model_fp)

models = [None, MLP.load(MLP_model_fp), MLP.load(MLP_model_fp_2), MLP.load(MLP_model_fp_3)]

training_models = []

################################################################################
################################################################################


def state_dict_list_to_tensor(state_dict):
    if state_dict is None: return None
    for key in state_dict.keys():
        state_dict[key] = torch.tensor(state_dict[key])
    return state_dict

def state_dict_tensor_to_list(state_dict):
    if state_dict is None: return None
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].tolist()
    return state_dict


app = dash.Dash(name=__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.index_string = open(html_fp, 'r').read()

app.layout = dbc.Container([
    dcc.Store("store", data={
        'img_label_pair': (None, None), 
        'new_img_n_clicks': 0, 
        'remove_img_n_clicks': 0,
        'train_new_model': False,
        'model_idx': 1,
        's': [784, 16, 10]
        }),

    dcc.Store("model_store"),

    dcc.Store("optimisation_status_store", data={
        'epoch': 0,
        'batch': 0,
        'loss': 0,
        'accuracy': 0,
        'epoch_completed': 0,
        'n_intervals': 0,
        'train_model_n_clicks': 0,
        }),
    dcc.Interval(id="check_training_status_interval", interval=10000, n_intervals=0),

    dbc.Row([
        # dbc.Col([
        #     dcc.Graph(id='digit_img'),
            
        # ], id='img_selection'),
        html.Div(dbc.Button(
            "New input",
            id="new_image_button",
            n_clicks=0,
            size="lg",  # Make the button large
            class_name="button"
        ), className="button_div"),
         html.Div(dbc.Button(
            "Remove input",
            id="remove_input_button",
            n_clicks=0,
            size="lg",  # Make the button large
            class_name="button"
        ), className="button_div"),
         html.Div(dbc.Button(
            "Train model",
            id="train_model_button",
            n_clicks=0,
            size="lg",  # Make the button large
            class_name="button"
        ), className="button_div"),
        html.Div(id="model_selection_div", children= [
            dcc.Dropdown(
                id = "model_selection_dropdown",
                options = [
                    {"label": "No model", "value": 0},
                    {'label': 'Model 1', 'value': 1},
                    {'label': 'Model 2', 'value': 2},
                    {'label': 'Model 3', 'value': 3},
                ],
                value = 1,
            )
        ])

    ]),

    dbc.Col([
        html.P("Input number of hidden layers"),
        dcc.Input(value = 1, id="hidden_layers_input", type="number", min=1, max=4, className="nn_param_input"),
        html.P("Input number of nodes in each hidden layer"),
        dbc.Row([
            dcc.Input(
                value = 16, id=f"n_nodes_{k}", type="number", min=1, max=128, className="n_nodes_input") 
                for k in range(1,5)
        ])
    ], id="nn_input_row"),
    
    dbc.Row([
        dcc.Graph(id='MLP_graph')
    ]),
    dbc.Row([
        dcc.Input(value = 1, id="n_epochs_input", type="number", min=1, max=10, className="nn_param_input"),
        dcc.Input(value = 64, id="batch_size_input", type="number", min=1, max=256, className="nn_param_input"),
        dcc.Input(value = 0.001, id="lr_input", type="number", min=0.001, max=1, step=0.001, className="nn_param_input"),
    ]),
    dbc.Row([
        html.P(id = "optimisation_status_text")
    ]),
    dbc.Row([
        dcc.Graph(id='opt_graph')
    ]),
])


@app.callback(
        Output("check_training_status_interval", 'interval'),
        [
            Input('train_model_button', 'n_clicks'),
            Input("n_epochs_input", 'value'),
            Input("batch_size_input", 'value'),
            Input("lr_input", 'value'),
         ],
         [
            State("store", "data"),
            State("optimisation_status_store", "data"),
         ]
)
def train_new_model(train_button_n_clicks, n_epochs, batch_size, lr, data, data_opt):
    if train_button_n_clicks <= data_opt['train_model_n_clicks']:
        return 10000
    
    # Train new model button has been clicked !

    # data['train_model_n_clicks'] = train_button_n_clicks
    # data['currently_training_network'] = True
    # data['n_batches'] = n_batches
    # data['batch_size'] = batch_size
    # data['lr'] = lr

    if len(training_models) > 1:
        return 500
    
    model = MLP(s=data['s'])
    process, queue = train_NN_process(model, n_epochs, batch_size, lr)
    training_models.append({'process': process, 'queue': queue})
    print(training_models[0]['process'].is_alive())
    models.append(model)
    return 500



@app.callback(
        Output("optimisation_status_text", "children"),
        Input("optimisation_status_store", "data")
)
def display_optimisation_progress(data):
    return f"Accuracy: {data['accuracy']:.2f}, loss: {data['loss']:.2f}, epoch: {data['epoch']} - {100*data['epoch_completed']:.2f}% complete"


@app.callback(
        [
            Output("optimisation_status_store", "data"),
            Output("model_selection_dropdown", "options")
        ],
        Input("check_training_status_interval", "n_intervals"),
        [
            State("model_selection_dropdown", "options"),
            State("optimisation_status_store", "data")
        ]
)
def check_training_progress(n_intervals, options, status_data):

    if len(training_models) == 0:
        return status_data, options
    if n_intervals <= status_data['n_intervals']:
        return status_data, options
    status_data['n_intervals'] = n_intervals

    training_model = training_models[0]
    process = training_model['process']
    queue = training_model['queue']
    if not queue.empty():
        queue_data = queue.get()
        for key in queue_data.keys():
            status_data[key] = queue_data[key]
    if not process.is_alive():
        training_models.pop()
        M = len(models)
        options.append({'label': f'Model {M-1}', 'value': M-1},)
    return status_data, options



@app.callback(
    Output("store", "data"),
    [
        Input("new_image_button", "n_clicks"), 
        Input("remove_input_button", "n_clicks"),
        Input("model_selection_dropdown", "value"),
        Input("hidden_layers_input", "value"),
        [Input("n_nodes_1", "value"),Input("n_nodes_2", "value"),Input("n_nodes_3", "value"),Input("n_nodes_4", "value")]
    ],
    State("store", "data")
)
def update_store(
    n_clicks_new, 
    n_clicks_remove, 
    model_idx,
    n_hidden_layers,
    n_nodes,
    data
    ):
    if n_clicks_new > data['new_img_n_clicks']:
        data['new_img_n_clicks'] = n_clicks_new
        data['img_label_pair'] = get_random_img_and_label()
    if n_clicks_remove > data['remove_img_n_clicks']:
        data['remove_img_n_clicks'] = n_clicks_remove
        data['img_label_pair'] = (None, None)
        
    data['model_idx'] = model_idx if model_idx is not None else 0

    data["s"] = ([784] + n_nodes)[:n_hidden_layers+1] + [10]
        
    return data



# @app.callback(
#     Output("digit_img", 'figure'),
#     Input("store", "data")
# )
# def update_digit_img(data):
#     img, _label = data['img_label_pair']
#     return display_img(img)






@app.callback(
        Output("MLP_graph", "figure"),
        Input("store", "data")
)
def update_MLP_graph(data):
    img, label = data['img_label_pair']
    if img is not None: img, label = map(torch.tensor, (img, label))
    # state_dict_list = data["state_dict"]
    # state_dict = state_dict_list_to_tensor(state_dict_list)
    model_idx = data['model_idx']
    fig = plot_NN(models[model_idx], input_img=img, s=data["s"])
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
