from re import template
from turtle import title, width
import dash  #(version 1.12.0)
from dash.dependencies import Input, Output
from dash import dash_table
from dash import dcc
from dash import html
import pandas as pd
import os
import torch
from math import dist
import matplotlib.patches as patches
import plotly.graph_objects as go
import random
random.seed(10)
import tkinter as TK
import plotly.express as px
from box_embeddings.modules.intersection import Intersection
from gensim.models import Word2Vec
import numpy as np
from sklearn.manifold import TSNE
    
def extract_embeddings(boxes):

                list_box = []
                list_box.append(boxes.z.data.tolist())
                list_box.append(boxes.Z.data.tolist())
                rect = patches.Rectangle((list_box[0][0], list_box[0][1]), 
                            list_box[1][0] - list_box[0][0], 
                            list_box[1][1] - list_box[0][1])
                
                rx, ry = rect.get_xy()
                cx = rx + rect.get_width()/2.0
                cy = ry + rect.get_height()/2.0

                a = (0.5,0.5)
                b = (cx,cy)

                distance = dist(a,b)

                return rect, [list_box[0][0], list_box[0][1], list_box[1][0], list_box[1][1]], cx, cy, distance

def most_similar(model, vocab, df, word, N=None):

        if N is None:
            N = len(vocab)

        embedding_all_target = model.embeddings_word.all_boxes

        
        index_word = (vocab[word])
        
        embedding_word = embedding_all_target[index_word]
        
        _, _, _, _, distance_word = extract_embeddings(embedding_word)

        volumes = model.box_vol(model.box_int(embedding_all_target, embedding_word))   

        idx = ((-volumes).argsort()).tolist()
        
        rows = []

        for i, index in enumerate(idx[0:N]):

            embedding_near = embedding_all_target[index]
            _, _, _, _, distance_near = extract_embeddings(embedding_near)

            rows.append([index, vocab.lookup_token(index), torch.exp(volumes[torch.tensor(index)]).item(),  
            torch.exp(model.box_vol(embedding_near)).item(), distance_near, df["Frequency"][index] ])
    

        df = pd.DataFrame(rows, columns=["Ix", "Most_Similar", "Volume_Int", "Volume_Target", "Distance_Target", "Frequency"])
    
        return df, idx[0:N]

def word_probability_similarity(word1, word2):

        embedding_all_target = model.embeddings_word.all_boxes

        index_word1 = (vocab[word1])
        index_word2 = (vocab[word2])

        word1 = embedding_all_target[index_word1]
        word2 = embedding_all_target[index_word2]

        score = model.box_vol(model.box_int(word1, word2))
        return torch.exp(score).item()


# -------------------------------------------------------------------------------------
# Import the cleaned data (importing csv into pandas)
folder = "weights/skipgram_WikiText2"
folder_ = "weights"
corpus = []
datasets = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fig = go.Figure()

fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])

# -------------------------------------------------------------------------------------
for filename in os.listdir(folder_):
    corpus.append({'label' : filename, 'value' : filename})


colors = []
df = pd.DataFrame()
df_correlations = pd.DataFrame()
model = None
vocab = None
word2vec_model = None


# App layout
app = dash.Dash(__name__, suppress_callback_exceptions=True) # this was introduced in Dash version 1.12.0

layout = go.Layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="#F0F0F0",
    xaxis={'showgrid': True},
    yaxis={'showgrid': True},
    template="ggplot2",
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family= 'italic'
    ),
    showlegend=False,
    )

fig = go.Figure(layout=layout)
fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])


app.layout = html.Div([


    html.Div([
            html.H2("Select corpus data", style={'text-align':'center', 'font-family': 'italic'}),
            dcc.Dropdown(
                id='corpus',
                options=corpus,
                value=corpus[0]["label"],
                clearable=False,
                className='dropdown',
                style={'text-align':'center', 'font-family': 'italic'},
            ),
            html.H2("Select set of hyper-parameters", style={'text-align':'center', 'font-family': 'italic'}),
            dcc.Dropdown(
                id='dataset',
                options= [],
                #value=datasets[0]["label"],
                clearable=False,
                className='dropdown',
                style={'text-align':'center', 'font-family': 'italic'},
            ),

            html.Br(),

            dcc.RadioItems(
                id='radio_button',
                style={'display': 'block' ,'font-style': 'italic', 'text-align':'center'},
                options=[
                    {'label': 'Final box space', 'value': 'final'},
                    {'label': 'Initial box space', 'value': 'init'}
                ],
                value='final')
            
        ]), 

    html.Div([
    html.H2('Box embedding Model vs Word2vec',  style={'text-align':'center', 'font-family':'italic'}),
    dcc.Tabs(id="tabs", value='explore', children=[
        dcc.Tab(label='Explore entire space', value='explore'),
        dcc.Tab(label='Calculate score probability', value='score'),
        dcc.Tab(label='View most similar', value='most_similar'),
        dcc.Tab(label='Analysis Box Embeddings', value='analysis'),
        dcc.Tab(label='Compare word2vec', value='word2vec'),
        dcc.Tab(label='Word2vec most_similar', value='word2vec_most'),
        dcc.Tab(label='Correlation results', value='results'),
    ]),

    html.Br(),

    html.Div(
        [
            dash_table.DataTable(
            id='datatable-interactivity',
            columns=[],
            data=[],  # the contents of the table
            editable=False,              # allow editing of data inside all cells
            filter_action="native",     # allow filtering of data by user ('native') or not ('none')
            sort_action="native",       # enables data to be sorted per-column by user or not ('none')
            sort_mode="single",         # sort across 'multi' or 'single' columns
            column_selectable="single",  # allow users to select 'multi' or 'single' columns
            row_selectable=None,     # allow users to select 'multi' or 'single' rows
            row_deletable=False,         # choose if user can delete a row (True) or not (False)
            selected_columns=[],        # ids of columns that user selects
            selected_rows=[],           # indices of rows that user selects
            page_action="native",       # all data is passed to the table up-front or not ('none')
            page_current=0,             # page number that user is on
            page_size=10,                # number of rows visible per page
            style_cell={                # ensure adequate header width when text is shorter than cell's text
                'minWidth': 95, 'maxWidth': 95, 'width': 95, 'font-family': 'italic',
            }
        ),
        html.Br(),

        dcc.ConfirmDialog(
        id='confirm-danger',
        message='Select exactly two elements!!',
        ),

        dcc.ConfirmDialog(
        id='confirm-danger_2',
        message='Select only one element!!',
        ),

        dcc.ConfirmDialog(
        id='confirm-danger_3',
        message='The plot of boxes works only in 2 dimension!!',
        ),

        dcc.ConfirmDialog(
        id='confirm-danger_4',
        message='Select at least two element to see them in the space!!',
        ),

        dcc.Tabs(id="tabs_spaces", value='target', children=[
            dcc.Tab(label='Target layer', value='target'),
            dcc.Tab(label='Context layer', value='context'),
        ]),

        html.Div([dcc.Graph(
            id='target_space',
            figure=fig,
        )], id='div_figure'),
        
    

        html.Div([
    
        
        ], id="space_analysis"),

        html.Br(),
        dash_table.DataTable(
            id='datatable-most',
            columns=[],
            data=[],  # the contents of the table
            editable=False,              # allow editing of data inside all cells
            filter_action="native",     # allow filtering of data by user ('native') or not ('none')
            sort_action="native",       # enables data to be sorted per-column by user or not ('none')
            sort_mode="single",         # sort across 'multi' or 'single' columns
            column_selectable="single",  # allow users to select 'multi' or 'single' columns
            row_selectable=None,     # allow users to select 'multi' or 'single' rows
            row_deletable=False,         # choose if user can delete a row (True) or not (False)
            selected_columns=[],        # ids of columns that user selects
            selected_rows=[],           # indices of rows that user selects
            page_action="native",       # all data is passed to the table up-front or not ('none')
            page_current=0,             # page number that user is on
            page_size=5,                # number of rows visible per page
            style_cell={                # ensure adequate header width when text is shorter than cell's text
                'minWidth': 95, 'maxWidth': 95, 'width': 95, 'font-family': 'italic'
            }
        ),

        ]),
        dash.html.H3("ecco lo score " ,style={'text-align':'center', 'font-family': 'italic'}, id="score"),
    html.Div(id='tabs-content-example-graph')
    ])

])

@app.callback(
    Output(component_id='dataset', component_property='value'),
    Output(component_id='dataset', component_property='options'),
    Input(component_id='corpus', component_property='value')
    
)
def update_(corpus):

    data = []
    
    for filename in os.listdir(folder_ + str("/") + str(corpus)):
        data.append({'label' : filename, 'value' : filename})

    return data[0]['label'], data

# -------------------------------------------------------------------------------------
@app.callback(
    Output('div_figure', 'style'),
    Output(component_id='space_analysis', component_property='children'),
    Output('tabs_spaces', 'children'),
    Output('datatable-most', 'columns'),
    Output('datatable-most', 'data'),
    Output(component_id='score', component_property='children'),
    Output('confirm-danger', 'displayed'),
    Output('confirm-danger_2', 'displayed'),
    Output('confirm-danger_3', 'displayed'),
    Output('confirm-danger_4', 'displayed'),
    Output('datatable-interactivity', 'row_selectable'),
    Output('datatable-interactivity', 'columns'),
    Output('datatable-interactivity', 'data'),
    Output(component_id='target_space', component_property='figure'),
    Output(component_id='datatable-interactivity', component_property='selected_rows'),
    Input('radio_button', 'value'),
    Input('tabs', 'value'),
    Input(component_id='corpus', component_property='value'),
    Input(component_id='dataset', component_property='value'),
    Input('datatable-interactivity', 'row_selectable'),
    Input(component_id='datatable-interactivity', component_property='selected_rows'),
    Input(component_id='tabs_spaces', component_property='value'),
    Input(component_id='score', component_property='children')
    
)
def update_dataset(radio_button, tabs, corpus, dataset, prop_row_selec, rows, tabs_spaces, scores):

    print(corpus)
    print(dataset)

    global df
    global layout 
    global model
    global vocab
    global word2vec_model
    global df_correlations
    global colors


    fig = go.Figure(layout=layout)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])


    fig.add_trace(go.Scatter(
        x=[0.5],
        y=[0.5],
        marker=dict(color="crimson", size=5),
        mode="markers",
    ))

    style_figure = {'display':'block'}

    if "embed_dim_2" in dataset:
        embed_dim_more_2 = True
    else:
        embed_dim_more_2 = False
        #### questa proprietà funzionerebbe solo che devo rendere possibile la selezione in word2vec
        #prop_row_selec = None

    
    folder = ("weights/" + str(corpus))

    if radio_button=="final": 
        df = pd.read_pickle(f'./{folder}/{dataset}/dataframe_final.pkl')        
        model = torch.load(f'{folder}/{dataset}/model_final.pt', map_location=device)
        vocab = torch.load(f'{folder}/{dataset}/vocab.pt')
        word2vec_model = Word2Vec.load(f'{folder}/{dataset}/word2vec.model')
        df_correlations = pd.read_pickle(f'./{folder}/{dataset}/dataframe_correlations.pkl')

        for x in range(0,len(vocab)):
            colors.append("#%06x" % random.randint(0, 0xFFFFFF))

    else:
        df = pd.read_pickle(f'./{folder}/{dataset}/dataframe_init.pkl')        
        model = torch.load(f'{folder}/{dataset}/model_init.pt', map_location=device)
        vocab = torch.load(f'{folder}/{dataset}/vocab.pt')
        word2vec_model = Word2Vec.load(f'{folder}/{dataset}/word2vec.model')
        df_correlations = pd.read_pickle(f'./{folder}/{dataset}/dataframe_correlations.pkl')

        for x in range(0,len(vocab)):
            colors.append("#%06x" % random.randint(0, 0xFFFFFF))


    print(rows)
    
    columns_most=[]
    data_most=[]
    columns=[]
    data=[]

    children_div_analysis = []

    children_ = [dcc.Tab(label='Target layer', value='target'), dcc.Tab(label='Context layer', value='context'),]

    message_error=False
    message_error_2=False
    message_error_3=False
    message_error_4=False

    if len(rows)>0 and embed_dim_more_2==False:
        message_error_3=True


    if tabs=="explore":
        scores = ""
        columns = [
            {"name": i, "id": i}
            for i in df.columns
            ]
        data = df.to_dict('records')
        prop_row_selec = "multi"

        if (tabs_spaces == 'target' and embed_dim_more_2):

            fig = go.Figure(layout=layout)
            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1])


            fig.add_trace(go.Scatter(
                x=[0.5],
                y=[0.5],
                marker=dict(color="crimson", size=5),
                mode="markers",
            ))
            for i, elem in enumerate(rows):

                _, p, cx, cy, distance = extract_embeddings(model.embeddings_word.all_boxes[elem])

                fig.add_trace(
                        go.Scatter(
                            x=[cx],
                            y=[cy],
                            mode="text",
                            text=[vocab.lookup_token(elem)],
                            #textposition="middle center",
                            textfont=dict(color="black"),
                            name=str([vocab.lookup_token(elem)][0]),
                            hovertemplate="<br>".join([
                                "x: " + str(cx),
                                "y: " + str(cy),
                                "vol: " + str(torch.exp(model.box_vol(model.embeddings_word.all_boxes[elem])).item()),
                                "distance: " + str(distance)
                            ]),
                        )
                )

                fig.add_shape(type="rect",
                    opacity = 0.2, fillcolor=colors[elem],
                    x0=p[0], y0=p[1], x1=p[2], y1=p[3],
                    line=dict(color="black",
                    width=2),
                )
        elif embed_dim_more_2:
            fig.add_trace(go.Scatter(
                x=[0.5],
                y=[0.5],
                marker=dict(color="crimson", size=5),
                mode="markers",
            ))
            for i, elem in enumerate(rows):

                _, p, cx, cy, distance = extract_embeddings(model.embeddings_context.all_boxes[elem])

                fig.add_trace(
                        go.Scatter(
                            x=[cx],
                            y=[cy],
                            mode="text",
                            text=[vocab.lookup_token(elem)],
                            textposition="middle center",
                            textfont=dict(color="black"),
                            name=str([vocab.lookup_token(elem)][0]),
                            hovertemplate="<br>".join([
                                "x: " + str(cx),
                                "y: " + str(cy),
                                "vol: " + str(torch.exp(model.box_vol(model.embeddings_context.all_boxes[elem])).item()),
                                "distance: " + str(distance)
                            ]),    
                        )
                )
            
                fig.add_shape(type="rect",
                    opacity = 0.2, fillcolor=colors[elem],
                    x0=p[0], y0=p[1], x1=p[2], y1=p[3],
                    line=dict(color="black", width=2),
                )
    elif tabs=="score":

        fig = go.Figure(layout=layout)
        fig.add_trace(go.Scatter(
                    x=[0.5],
                    y=[0.5],
                    marker=dict(color="crimson", size=5),
                    mode="markers",
                ))
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        columns = [
            {"name": i, "id": i}
            for i in df.columns
            ]
        data = df.to_dict('records')
        prop_row_selec = "multi"
        
        if len(rows)<2:
            message_error = True
            scores = ""
        elif len(rows)>2:
            message_error = True
            scores = ""
        else:
            if tabs_spaces == 'target' and embed_dim_more_2:

                fig.add_trace(go.Scatter(
                    x=[0.5],
                    y=[0.5],
                    marker=dict(color="crimson", size=5),
                    mode="markers",
                ))
                for i, elem in enumerate(rows):

                    _, p, cx, cy, distance = extract_embeddings(model.embeddings_word.all_boxes[elem])

                    fig.add_trace(
                            go.Scatter(
                                x=[cx],
                                y=[cy],
                                mode="text",
                                text=[vocab.lookup_token(elem)],
                                #textposition="middle center",
                                textfont=dict(color="black"),
                                name=str([vocab.lookup_token(elem)][0]),
                                hovertemplate="<br>".join([
                                    "x: " + str(cx),
                                    "y: " + str(cy),
                                    "vol: " + str(torch.exp(model.box_vol(model.embeddings_word.all_boxes[elem])).item()),
                                    "distance: " + str(distance)
                                ]),
                            )
                    )

                    fig.add_shape(type="rect",
                        opacity = 0.2, fillcolor=colors[elem],
                        x0=p[0], y0=p[1], x1=p[2], y1=p[3],
                        line=dict(color="black",
                        width=2),
                    )

                #### CALCULATE SIMILARITY SCORE ####
                
                scores = word_probability_similarity(vocab.lookup_token(rows[0]), vocab.lookup_token(rows[1]))
                scores = ("The probability score between " + vocab.lookup_token(rows[0]) + " and " + vocab.lookup_token(rows[1]) + " is: " + str(scores))
            elif embed_dim_more_2:

                fig.add_trace(go.Scatter(
                    x=[0.5],
                    y=[0.5],
                    marker=dict(color="crimson", size=5),
                    mode="markers",
                ))
                for i, elem in enumerate(rows):

                    _, p, cx, cy, distance = extract_embeddings(model.embeddings_context.all_boxes[elem])

                    fig.add_trace(
                            go.Scatter(
                                x=[cx],
                                y=[cy],
                                mode="text",
                                text=[vocab.lookup_token(elem)],
                                textposition="middle center",
                                textfont=dict(color="black"),
                                name=str([vocab.lookup_token(elem)][0]),
                                hovertemplate="<br>".join([
                                    "x: " + str(cx),
                                    "y: " + str(cy),
                                    "vol: " + str(torch.exp(model.box_vol(model.embeddings_context.all_boxes[elem])).item()),
                                    "distance: " + str(distance)
                                ]),    
                            )
                    )
                
                    fig.add_shape(type="rect",
                        opacity = 0.2, fillcolor=colors[elem],
                        x0=p[0], y0=p[1], x1=p[2], y1=p[3],
                        line=dict(color="black", width=2),
                    )
    elif tabs=="most_similar":

        fig = go.Figure(layout=layout)
        fig.add_trace(go.Scatter(
                    x=[0.5],
                    y=[0.5],
                    marker=dict(color="crimson", size=5),
                    mode="markers",
                ))
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        columns = [
            {"name": i, "id": i}
            for i in df.columns
            ]
        data = df.to_dict('records')
        prop_row_selec = "multi"

        if len(rows)<1:
            message_error_2 = True
            scores = ""
        elif len(rows)>1:
            message_error_2 = True
            scores = ""
        else:
            dataf = pd.DataFrame()            
            dataf, idx = most_similar(model, vocab, df, vocab.lookup_token(rows[0]), N=10)


            columns_most = [
            {"name": i, "id": i}
            for i in dataf.columns
            ]
            data_most = dataf.to_dict('records')


            if tabs_spaces == 'target' and embed_dim_more_2:

                fig.add_trace(go.Scatter(
                    x=[0.5],
                    y=[0.5],
                    marker=dict(color="crimson", size=5),
                    mode="markers",
                ))

                _, p, cx, cy, distance = extract_embeddings(model.embeddings_word.all_boxes[rows[0]])

                fig.add_trace(
                        go.Scatter(
                            x=[cx],
                            y=[cy],
                            mode="text",
                            text=[vocab.lookup_token(rows[0])],
                            #textposition="middle center",
                            textfont=dict(color="black"),
                            name=str([vocab.lookup_token(rows[0])][0]),
                            hovertemplate="<br>".join([
                                "x: " + str(cx),
                                "y: " + str(cy),
                                "vol: " + str(torch.exp(model.box_vol(model.embeddings_word.all_boxes[rows[0]])).item()),
                                "distance: " + str(distance)
                            ]),
                        )
                )

                fig.add_shape(type="rect",
                    opacity = 0.2, fillcolor=colors[rows[0]],
                    x0=p[0], y0=p[1], x1=p[2], y1=p[3],
                    line=dict(color="black",
                    width=2),
                )


                for i, elem in enumerate(idx):

                    _, p, cx, cy, distance = extract_embeddings(model.embeddings_word.all_boxes[elem])

                    fig.add_trace(
                            go.Scatter(
                                x=[cx],
                                y=[cy],
                                mode="text",
                                text=[vocab.lookup_token(elem)],
                                #textposition="middle center",
                                textfont=dict(color="black"),
                                name=str([vocab.lookup_token(elem)][0]),
                                hovertemplate="<br>".join([
                                    "x: " + str(cx),
                                    "y: " + str(cy),
                                    "vol: " + str(torch.exp(model.box_vol(model.embeddings_word.all_boxes[elem])).item()),
                                    "distance: " + str(distance)
                                ]),
                            )
                    )

                    fig.add_shape(type="rect",
                        opacity = 0.2, fillcolor=colors[elem],
                        x0=p[0], y0=p[1], x1=p[2], y1=p[3],
                        line=dict(color="black",
                        width=2),
                    )
            elif embed_dim_more_2:

                fig.add_trace(go.Scatter(
                    x=[0.5],
                    y=[0.5],
                    marker=dict(color="crimson", size=5),
                    mode="markers",
                ))

                _, p, cx, cy, distance = extract_embeddings(model.embeddings_context.all_boxes[rows[0]])

                fig.add_trace(
                        go.Scatter(
                            x=[cx],
                            y=[cy],
                            mode="text",
                            text=[vocab.lookup_token(rows[0])],
                            #textposition="middle center",
                            textfont=dict(color="black"),
                            name=str([vocab.lookup_token(rows[0])][0]),
                            hovertemplate="<br>".join([
                                "x: " + str(cx),
                                "y: " + str(cy),
                                "vol: " + str(torch.exp(model.box_vol(model.embeddings_context.all_boxes[rows[0]])).item()),
                                "distance: " + str(distance)
                            ]),
                        )
                )

                fig.add_shape(type="rect",
                    opacity = 0.2, fillcolor=colors[rows[0]],
                    x0=p[0], y0=p[1], x1=p[2], y1=p[3],
                    line=dict(color="black",
                    width=2),
                )


                for i, elem in enumerate(idx):

                    _, p, cx, cy, distance = extract_embeddings(model.embeddings_context.all_boxes[elem])

                    fig.add_trace(
                            go.Scatter(
                                x=[cx],
                                y=[cy],
                                mode="text",
                                text=[vocab.lookup_token(elem)],
                                textposition="middle center",
                                textfont=dict(color="black"),
                                name=str([vocab.lookup_token(elem)][0]),
                                hovertemplate="<br>".join([
                                    "x: " + str(cx),
                                    "y: " + str(cy),
                                    "vol: " + str(torch.exp(model.box_vol(model.embeddings_context.all_boxes[elem])).item()),
                                    "distance: " + str(distance)
                                ]),    
                            )
                    )
                
                    fig.add_shape(type="rect",
                        opacity = 0.2, fillcolor=colors[elem],
                        x0=p[0], y0=p[1], x1=p[2], y1=p[3],
                        line=dict(color="black", width=2),
                    )
    elif tabs=="analysis":

        prop_row_selec=None
        children_ = []
        dataff = pd.DataFrame({"frequency": list(df["Frequency"].values)})
        dataff = dataff.groupby(['frequency']).size().reset_index(name='counts')

        fig = px.bar(dataff, x="frequency", y="counts", title="Frequency distribution of words", log_x=True)


        fig1 = go.Figure(layout=layout)
        fig1.update_xaxes(range=[0, 1])
        fig1.update_yaxes(range=[0, 1])

        if embed_dim_more_2:
            children_div_analysis = [
                dcc.Tabs(id="tabs_spaces_analysis", value='target_analysis', children=[
                dcc.Tab(label='Target layer', value='target_analysis'),
                dcc.Tab(label='Context layer', value='context_analysis'),
            ]),

            dcc.Graph(
                id='target_space_target',
                figure=fig1,
            ),

            dcc.Tabs(id="tabs_spaces_analysis_distance", value='target_analysis_distance', children=[
                dcc.Tab(label='Target layer', value='target_analysis_distance'),
                dcc.Tab(label='Context layer', value='context_analysis_distance'),
            ]),

            dcc.Graph(
                id='target_space_target_distance',
                figure=fig1,
            )
            
            ]
    elif tabs=="word2vec":

        fig = go.Figure(layout=layout)

        fig.add_trace(go.Scatter(
                x=[0.5],
                y=[0.5],
                marker=dict(color="crimson", size=5),
                mode="markers",
            ))

        scores = ""
        columns = [
            {"name": i, "id": i}
            for i in df.columns
            if i in ["Ix", "Word", "Frequency"]
            ]
        data = df.to_dict('records')
        prop_row_selec = "multi"

        children_=[]
        
        keyed = word2vec_model.wv

        embeddings = []
        words = []
        for i, elem in enumerate(rows):

            word = vocab.lookup_token(elem)
            words.append(word)
            word_idx = word2vec_model.wv.key_to_index[word]
            print(word_idx)
            
            embeddings.append(keyed.get_vector(word_idx, norm=True))

        if len(embeddings)<=1:
            message_error_4 = True

        if len(embeddings)>1:

            embeddings_df = pd.DataFrame(embeddings)
            # t-SNE transform
            tsne = TSNE(n_components=2)
            embeddings_df_trans = tsne.fit_transform(embeddings_df)
            embeddings_df_trans = pd.DataFrame(embeddings_df_trans)

            # get token order
            embeddings_df_trans.index = words

  
            fig.add_trace(
                go.Scatter(
                    x=embeddings_df_trans[0],
                    y=embeddings_df_trans[1],
                    mode="text",
                    text=embeddings_df_trans.index,
                    textposition="middle center",
                    textfont=dict(color="black"),
                )
            )
            
    elif tabs=="word2vec_most":
        print("ciao")
        scores = ""
        columns = [
            {"name": i, "id": i}
            for i in df.columns
            if i in ["Ix", "Word", "Frequency"]
            ]
        data = df.to_dict('records')
        prop_row_selec = "multi"

        children_=[]

        if len(rows)<1:
            message_error_2 = True
            scores = ""
        elif len(rows)>1:
            message_error_2 = True
            scores = ""
        else:
            
            word_idx = word2vec_model.wv.key_to_index
            print(len(word_idx))
            print(len(vocab))
            rows_ = []

            word = vocab.lookup_token(rows[0])
            similars = word2vec_model.wv.most_similar(positive=[word], topn=20)

            for i, element in enumerate(similars):

                indic = vocab.lookup_indices([element[0]])[0]
                
                rows_.append([ indic, element[0], element[1], df["Frequency"][indic] ])
                    
            try:
                dataf = pd.DataFrame(rows_, columns=["Ix", "Word_similar", "Score", "Frequency"])
            except Exception as e: print(e)

            columns_most = [
            {"name": i, "id": i}
            for i in dataf.columns
            ]
            
            data_most = dataf.to_dict('records')

            keyed = word2vec_model.wv

            embeddings = []
            words = []

            word = vocab.lookup_token(rows[0])
           
            words.append(word)
            word_idx = word2vec_model.wv.key_to_index[word]
    
            
            embeddings.append(keyed.get_vector(word_idx, norm=True))

            for i, element in enumerate(similars):

                word = vocab.lookup_token(vocab.lookup_indices([element[0]])[0])
                print(word)
                print(element[0])
                words.append(word)
                word_idx = word2vec_model.wv.key_to_index[word]
                print(word_idx)
                
                embeddings.append(keyed.get_vector(word_idx, norm=True))

            embeddings_df = pd.DataFrame(embeddings)
            # t-SNE transform
            tsne = TSNE(n_components=2)
            embeddings_df_trans = tsne.fit_transform(embeddings_df)
            embeddings_df_trans = pd.DataFrame(embeddings_df_trans)

            # get token order
            embeddings_df_trans.index = words

            fig = go.Figure(layout=layout)

            fig.add_trace(go.Scatter(
                x=[0.5],
                y=[0.5],
                marker=dict(color="crimson", size=5),
                mode="markers",
            ))

            fig.add_trace(
                go.Scatter(
                    x=embeddings_df_trans[0],
                    y=embeddings_df_trans[1],
                    mode="text",
                    text=embeddings_df_trans.index,
                    textposition="middle center",
                    textfont=dict(color="black"),
                )
            )
    elif tabs == 'results':
        print("ciao")
        scores = ""
        columns = [
            {"name": i, "id": i}
            for i in df_correlations.columns
            ]
        data = df_correlations.to_dict('records')
        prop_row_selec = None

        children_=[]
        style_figure = {'display':'none'}
        


    return  style_figure, children_div_analysis, children_, columns_most, data_most, scores, message_error, message_error_2, message_error_3, message_error_4, prop_row_selec, columns, data, fig, rows


@app.callback(
    Output(component_id='target_space_target', component_property='figure'),
    Output(component_id='target_space_target_distance', component_property='figure'),
    Input(component_id='tabs_spaces_analysis', component_property='value'),
    Input(component_id='tabs_spaces_analysis_distance', component_property='value')
)

def update_dataset(tabs_value_analysis, tabs_value_analysis_distance):

    lis_all = list(range(len(vocab)))

    fig1 = go.Figure(layout=layout)
    fig1.update_xaxes(range=[0, 1])
    fig1.update_yaxes(range=[0, 1])

    fig1.add_trace(go.Scatter(
                    x=[0.5],
                    y=[0.5],
                    marker=dict(color="crimson", size=5),
                    mode="markers",
                ))

    fig2 = go.Figure(layout=layout)
    fig2.update_xaxes(range=[0, 1])
    fig2.update_yaxes(range=[0, 1])

    fig2.add_trace(go.Scatter(
                    x=[0.5],
                    y=[0.5],
                    marker=dict(color="green", size=5),
                    mode="markers",
                ))

    boxes_target = model.embeddings_word.all_boxes
    boxes_context = model.embeddings_context.all_boxes


    if tabs_value_analysis=="target_analysis":

        
        for i in lis_all:
            emb = boxes_target[i]
            _, p, cx, cy, distance = extract_embeddings(emb)

            fig1.add_trace(
                go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="text",
                    text=[vocab.lookup_tokens([i])[0]],
                    textposition="middle center",
                    textfont=dict(color="black"),
                    name=str([vocab.lookup_token(i)][0]),
                    hovertemplate="<br>".join([
                        "word: " + str([vocab.lookup_token(i)][0]),
                        "x: " + str(cx),
                        "y: " + str(cy),
                        "vol: " + str(torch.exp(model.box_vol(boxes_target[i])).item()),
                        "distance: " + str(distance)
                    ])
                )
            )

    else:

        for i in lis_all:
            emb = boxes_context[i]
            _, p, cx, cy, distance = extract_embeddings(emb)

            fig1.add_trace(
                    go.Scatter(
                        x=[cx],
                        y=[cy],
                        mode="text",
                        text=[vocab.lookup_tokens([i])[0]],
                        textposition="middle center",
                        textfont=dict(color="black"),
                        name=str([vocab.lookup_token(i)][0]),
                        hovertemplate="<br>".join([
                            "word: " + str([vocab.lookup_token(i)][0]),
                            "x: " + str(cx),
                            "y: " + str(cy),
                            "vol: " + str(torch.exp(model.box_vol(boxes_context[i])).item()),
                            "distance: " + str(distance)
                        ])
                    )
                )

    if tabs_value_analysis_distance=="target_analysis_distance":

        for i in lis_all:
            emb = boxes_target[i]
            _, p, cx, cy, distance = extract_embeddings(emb)
    
            fig2.add_trace(
                go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="markers",
                    marker=dict(color="crimson", size=5),
                    textposition="middle center",
                    textfont=dict(color="black"),
                    name=str([vocab.lookup_token(i)][0]),
                    hovertemplate="<br>".join([
                        "word: " + str([vocab.lookup_token(i)][0]),
                        "x: " + str(cx),
                        "y: " + str(cy),
                        "vol: " + str(torch.exp(model.box_vol(boxes_target[i])).item()),
                        "distance: " + str(distance)
                    ])
                )  
            )

    else:

        for i in lis_all:
            emb = boxes_context[i]
            _, p, cx, cy, distance = extract_embeddings(emb)

            fig2.add_trace(
                    go.Scatter(
                        x=[cx],
                        y=[cy],
                        mode="markers",
                        marker=dict(color="crimson", size=5),
                        textposition="middle center",
                        textfont=dict(color="black"),
                        name=str([vocab.lookup_token(i)][0]),
                        hovertemplate="<br>".join([
                            "word: " + str([vocab.lookup_token(i)][0]),
                            "x: " + str(cx),
                            "y: " + str(cy),
                            "vol: " + str(torch.exp(model.box_vol(boxes_context[i])).item()),
                            "distance: " + str(distance)
                        ])
                    )  
                )

    return fig1, fig2


if __name__ == '__main__':
    app.run_server(debug=True)