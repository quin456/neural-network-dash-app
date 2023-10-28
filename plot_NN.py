import torch 
torch.set_num_threads(2)
from torch import nn
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from digit_classifier_NN import (
    MLP,
    MLP_model_fp,
    MLP_model_fp_2,
    train_images, 
    test_images, 
    train_labels, 
    test_labels,
    get_s_from_state_dict,
    get_keys,
    train_NN_process
)


unit = 0.02

node_color = 'lightblue'
node_marker_size = 12

min_edge_weight = 0.6

def get_random_img_and_label(images=test_images, labels=test_labels):
    idx = np.random.randint(0,len(images)-1)
    return images[idx], labels[idx]

def remove_axes(fig):
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False)
        )
    return fig

def display_img(img):
    if img is None:
        return remove_axes(go.Figure())
    fig =  px.imshow(img, color_continuous_scale='gray')
    remove_axes(fig)
    return fig

def get_digit_img_display(images=test_images):
    img, _label = get_random_img_and_label(images)
    return display_img(img)


def get_square_input_layer(s0, x0):
    if s0 != 784:
        raise Exception(f"Detected invalid s[0] value: {s0}")
    h = 28
    w = 28

    sep = 1*unit
    y0 = 28*sep/2

    locs = np.zeros((h,w,2))
    for i in range(h):
        for j in range(w):
            locs[j,i,0] = x0 - (h-i)*sep
            locs[j,i,1] = y0 - j*sep
    return locs.reshape(784,2)

def get_node_positions(s, input_nodes):
    L = len(s)
    h_sep = [10*unit, 20*unit, 25*unit, 25*unit, 25*unit, 25*unit]
    v_sep = [-1*unit]*L
    node_locs = []
    for layer in range(L):

        if input_nodes and layer==0:
            node_locs.append(get_square_input_layer(s[0], -h_sep[0]))
            continue

        top = (s[layer]-1)*v_sep[layer]
        bottom = -top

        x = layer*h_sep[layer]*np.ones(s[layer])
        y = np.linspace(bottom, top, s[layer])
        node_locs.append(np.stack((x,y)).T)

    return node_locs



def draw_node(fig, pos, radius=1*unit, line_color = 'black', fill_color = 'lightblue'):

    x,y = pos 
    # Add a circle
    fig.add_shape(
        go.layout.Shape(
            type="circle",
            xref="x",
            yref="y",
            x0=x-radius,
            y0=y-radius,
            x1=x+radius,
            y1=y+radius,
            line=dict(color=line_color),
            fillcolor = fill_color
        )
    )

    # Update axes properties
    fig.update_xaxes(range=[-1, 3])
    fig.update_yaxes(range=[-1, 3])

def normalise_to_1(A, minimum = None):
    package = torch if type(A) == torch.Tensor else np
    if minimum is None: minimum = package.min(A)
    return (A - minimum) / (package.max(A) - minimum)

def get_random_node_colors(s):
    color = []
    for k in s:
        color.append(np.random.rand(k))
    return color


def node_colors_from_vals(node_values):
    color = []

    for nv in node_values:
        layer_colors = normalise_to_1(nv[0])
        layer_colors = node_cmap(normalise_to_1(nv[0]))
        layer_colors_formatted = [0]*len(layer_colors)
        for j in range(len(layer_colors)):
            layer_colors_formatted[j] = to_dumb_format(layer_colors[j])
        color.append(layer_colors_formatted)
    return color



def draw_nodes(fig, s, node_locs, color=node_color, node_vals=None):
    if node_vals is not None:
        color=node_colors_from_vals(node_vals)
    else:
        color = [[color]*sl for sl in s]

    for k,node_loc_layer in enumerate(node_locs):
        fig.add_trace(go.Scatter(
            x=node_loc_layer[:,0], 
            y=node_loc_layer[:,1],
            mode='markers',
            marker=dict(color=color[k], size=node_marker_size,
            line=dict(
            width=1,
            color='black'
        ))
        ))
        # for pos in node_loc_layer:
        #     draw_node(fig,pos)]

def to_dumb_format(rgba):
    rgb = (rgba[:-1]*255).astype(int)
    return f'rgb({rgb[0]},{rgb[1]},{rgb[2]},0.1)'

def node_cmap(z):
    rgba = np.array(plt.get_cmap("YlGn")(z))
    return rgba

def edge_cmap(z):
    if type(z) == torch.Tensor: z = z.item() 
    rgba = np.array(plt.get_cmap("YlGn")(z))
    return to_dumb_format(rgba)

def get_edge_weights(state_dict, node_vals, L, input_nodes, img):
    edge_weight = []

    keys = get_keys(L, input_nodes)
    for l, key in enumerate(keys):
        weights = state_dict[key]
        weights = normalise_to_1(weights)
        weights = normalise_to_1(weights, min_edge_weight)
        
        if node_vals is None:
            edge_weight.append(np.array(weights).T)
            continue
        n,m = weights.shape
        layer_colors = [n*[0] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                layer_colors[i][j] = weights[j][i].item()*node_vals[l][0][i]
        edge_weight.append(layer_colors)
    return edge_weight


def draw_edges(fig, state_dict, node_locs, node_vals, L, input_nodes, img):

    edge_weight = get_edge_weights(state_dict, node_vals, L, input_nodes, img)
    keys = get_keys(L, input_nodes)
    for l, key in enumerate(keys):
        weights = state_dict[key]
        n,m = weights.shape
        for i in range(m):
            for j in range(n):
                weight_lij = edge_weight[l][i][j]
                if weight_lij < 0.01:
                    continue
                fig.add_trace(go.Scatter(
                    x=[node_locs[l][i][0], node_locs[l+1][j][0]], 
                    y=[node_locs[l][i][1], node_locs[l+1][j][1]], 
                    mode='lines',
                    line = dict(color = edge_cmap(weight_lij), width=0.7)
                    ))






def get_untrained_state_dict(s, input_nodes):
    L = len(s)
    keys = get_keys(L, input_nodes)
    state_dict = {}
    l = 0 if input_nodes else 1
    for key in keys:
        state_dict[key] = torch.ones((s[l+1], s[l]))*0.5
        l += 1
    return state_dict


def plot_NN(model, input_img=None, s=[784, 16, 10], input_nodes=True):

    fig = go.Figure()

    if model is not None:
        state_dict = model.state_dict()
        s = get_s_from_state_dict(state_dict)
    else:
        state_dict = get_untrained_state_dict(s, input_nodes)

    L = len(s)
    if not input_nodes: s = s[1:]
    node_locs = get_node_positions(s, input_nodes = input_nodes)
    node_vals = model.get_node_vals(input_img, input_nodes) if model is not None else None
    if model is not None:
        draw_edges(fig, state_dict, node_locs, node_vals, L, input_nodes, input_img)
    draw_nodes(fig, s, node_locs, node_vals=node_vals)
                
    fig.update_layout(showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    remove_axes(fig)
    return fig


def plot_opt_progress():
    fig = go.Figure()
    


    return fig




def test_plot_opt_progress():
    model = MLP(s=[784, 16, 10])
    process, queue = train_NN_process(model, n_epochs=1, batch_size=64, lr=0.001)

    data0 = {
        'epoch': 0,
        'batch': 0,
        'loss': 0,
        'accuracy': 0,
        'epoch_completed': 0,
        'n_intervals': 0,
        'train_model_n_clicks': 0,
    },

    while process.is_alive():
        if not queue.empty():
            status = queue.get()
            print(status)
    process.join()

if __name__ == '__main__':
    # model = MLP.load(MLP_model_fp_2)
    model = MLP(s=[784, 16, 16, 16, 10])
    train_NN_process(model, n_epochs=5, batch_size=100, lr=0.001)
    plot_NN(model=model, input_img=get_random_img_and_label()[0], input_nodes=True).show()
    # display_img(None).show()
    # fig = go.Figure()
    # draw_node(fig, (1,1), 1)
    # fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # fig.show()