import dash
from dash import dcc, html,dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import datetime as dt
import base64
import plotly.express as px
import geopandas as gpd
from sklearn.preprocessing import robust_scale
from sklearn.cluster import KMeans
import requests
import os
import json
from loguru import logger
import matplotlib as plt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app server
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# PREDICTION API URL 
api_url = os.getenv('API_URL')
api_url = "http://3.94.52.90:8001/api/v1/predict".format(api_url)


with open('../data/cultivos.csv', 'r', encoding='utf-8') as file:
    Inputs = pd.read_csv(file)

municipios_mapa=gpd.read_file('../data/MunicipiosVeredas19MB.json')


grupos_cultivos = Inputs['GRUPO_CULTIVO'].unique().tolist()
años = Inputs['ANIO'].unique().tolist()
departamentos = Inputs['NOMBRE_DEPARTAMENTO'].unique().tolist()
cultivos = Inputs['NOMBRE_CULTIVO'].unique().tolist()
Clusters = Inputs['NUM_CLUSTERS'].unique()

CantidadCultivos = len(Inputs['NOMBRE_CULTIVO'].unique())
NumClusters = len(Inputs['NUM_CLUSTERS'].unique())
valores_unicos = list(range(1, NumClusters + 1))


NumMunicipios = len(Inputs['NOMBRE_MUNICIPIO'].unique())
NumMunicipios_formateado = "{:,}".format(NumMunicipios)

# Datos de ejemplo Tabla

df = pd.DataFrame(Inputs, columns=['GRUPO_CULTIVO','NOMBRE_CULTIVO','NUM_CLUSTERS', 'RENDIMIENTO_TONELADAS_HA'])
df_top10 = df.apply(lambda x: x.unique()[:10])

#########################DASH##########################                                                                                                                
with open('./img/Logo.png', 'rb') as f:
    logo_data = f.read()
encoded_logo = base64.b64encode(logo_data).decode()

with open('./img/mapa.png', 'rb') as f:
    mapa_data = f.read()
encoded_mapa = base64.b64encode(mapa_data).decode()


app.layout = html.Div([
# Contenedor header
html.Div([
    html.Img(src=f"data:image/png;base64,{encoded_logo}", height=100),
    html.H1("Rendimiento Agrícola por Hectárea"),
], className="header"),

#contenedor de los dropdowns o filtros
html.Div([
# Contenedor fILTROS
html.Div([
html.H6("Departamento"),
    dcc.Dropdown(
        id="dropdown-departamento",
        options=[{'label': departamento, 'value': departamento} for departamento in departamentos],
        value=departamentos[0],
        searchable=True  # Habilitar la opción de búsqueda
    ),
html.H6("Municipio"),
    dcc.Dropdown(
        id="dropdown-municipio",
        # Aquí dejamos las opciones vacías por ahora, se llenarán con la actualización
        options=[],
        value=None,
        searchable=True  # Habilitar la opción de búsqueda
    ),
html.H6("Grupo de Cultivo"),
    dcc.Dropdown(
        id="dropdown-grupo-cultivo",
        options=[],
        value=None,
        searchable=True  # Habilitar la opción de búsqueda
    ),
html.H6("Número de Cluster"),
     dcc.Dropdown(
        id="dropdown-numero-cluster",
        options=[],
        value=None,
        searchable=True  # Habilitar la opción de búsqueda
    ),

html.H6("Año"),
    dcc.Dropdown(
        id="dropdown-año",
        options=[{'label': str(año), 'value': año} for año in años],
        value=años[0],
        searchable=True  # Habilitar la opción de búsqueda
    ),

html.H6("Cultivo"),
    dcc.Dropdown(
        id="dropdown-cultivo",
        options=[{'label': cultivo, 'value': cultivo} for cultivo in cultivos],
        value=cultivos[0],
        searchable=True  # Habilitar la opción de búsqueda
    ),

html.Button("Enviar Consulta",
    id="enviar-button",
    n_clicks=0,
    style={'color': 'white'}
    ),
     
html.Br(),
    html.H6(html.Div(id='resultado')),


], className="left-container"),  # Contenedor izquierdo

html.Div([
html.Div([             
# Contenedor cards            

html.Div([             
    # Contenedor cards
    html.Div([                  
        html.Div("Cantidad de Cultivos", className="card-title"),
        html.Div(f"{CantidadCultivos}", className="card-valor", 
                 style={'backgroundColor': 'rgba(0,0,0,0)', 'color': '#2cfec1', 'textAlign': 'center','fontSize': '30px'}
                 ),
                 
    ], className="card"),           

    html.Div([
        html.Div("Número de Clusters", className="card-titdle"),
        html.Div(f"{NumClusters}", className="card-valor", 
                 style={'backgroundColor': 'rgba(0,0,0,0)', 'color': '#2cfec1', 'textAlign': 'center','fontSize': '30px'}
                 ),
    ], className="card"),

    html.Div([
        html.Div("Total de Municipios", className="card-title"),
        html.Div(f"{NumMunicipios_formateado}", className="card-valor", 
                 style={'backgroundColor': 'rgba(0,0,0,0)', 'color': '#2cfec1', 'textAlign': 'center','fontSize': '30px'}
                 ),
    ], className="card"),# Contenedor cards
    # Contenedor de la tabla
    html.Div([
        html.H4('Tabla de Recomendaciones', className="title-visualizacion"),
        dash_table.DataTable(df_top10.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
                             style_cell={'backgroundColor': 'rgba(0,0,0,0)', 'color': '#2cfec1','textAlign': 'center'},
                             ), 
    ], className="table-container"),  # Contenedor de la tabla
], className="cards-container"), # Contenedor de las cards y la tabla

# Contenedor de la visualización del mapa
html.Div([
    html.H4('Grupo de Cultivos con Municipicos Similares', className="title-visualizacion"),
    html.Img(src=f"data:image/png;base64,{encoded_mapa}"),
 
],className="mapa-container"),
# Contenedor de la tabla

html.Div([
    html.H4('Predicción del Rendimiento en Toneladas por Hectarea', className="title-visualizacion"),
    html.Div(id='resultado', className="card-valor", 
                 style={'backgroundColor': 'rgba(0,0,0,0)', 'color': '#2cfec1', 'textAlign': 'center','fontSize': '50px'}
                 ),
        dcc.Graph(id="line-chart", className="line-chart"),
    ], className="line-chart-container"),


], className="cards-container"), # Contenedor de las cards y la tabla
], className="main-container"),  # Contenedor derecho
#contenedor de las cards y la tabla


# Contenedor de la visualización del mapa

], className="main-container")
# Contenedor de la visualización de la gráfica de líneas  

])


#Llamado a la base de datosropdown municipio
@app.callback(
    Output("dropdown-municipio", "options"),
    [Input("dropdown-departamento", "value")]
)
def update_municipios(departamento):
    # Filtra los municipios basados en el departamento seleccionado
    municipios = Inputs[Inputs['NOMBRE_DEPARTAMENTO'] == departamento]['NOMBRE_MUNICIPIO'].unique()
    options = [{'label': municipio, 'value': municipio} for municipio in municipios]
    return options

#Llamado a la base de datosropdown grupocultivo
@app.callback(
    Output("dropdown-grupo-cultivo", "options"),
    [Input("dropdown-municipio", "value")]
)
def update_municipios(municipios):
    # Filtra los municipios basados en el departamento seleccionado
    grupos_cultivos = Inputs[Inputs['NOMBRE_MUNICIPIO'] == municipios]['GRUPO_CULTIVO'].unique()
    options = [{'label': grupo, 'value': grupo} for grupo in grupos_cultivos]
    return options

#Llamado a la base de datosropdown cluster
@app.callback(
    Output("dropdown-numero-cluster", "options"),
    [Input("dropdown-grupo-cultivo", "value")]
)
def update_municipios(grupos_cultivos):
    # Filtra los municipios basados en el departamento seleccionado
    grupos_cultivos = Inputs[Inputs['GRUPO_CULTIVO'] == grupos_cultivos]['NUM_CLUSTERS'].unique()
    options = [{'label': str(cluster), 'value': cluster} for cluster in Clusters]
    
    return options

##LLammado boton 
from dash.exceptions import PreventUpdate

# ...

#Llamado a la base de card 1

#llamado a la base de tabla


##mapa #https://plotly.com/python/maps/



##line chart
@app.callback(
    Output("line-chart", "figure"),
    [Input("dropdown-grupo-cultivo", "value"),
     Input("dropdown-año", "value"),
     Input("dropdown-municipio", "value"),
     Input("dropdown-departamento", "value"),
     Input("dropdown-numero-cluster", "value"),
     Input("dropdown-cultivo", "value")]

)

def update_line_chart(grupo_cultivo, año, municipio, departamento, cultivo,valores_unicos):
    # Filtra los datos basados en las selecciones
    filtered_data = Inputs[
        (Inputs['GRUPO_CULTIVO'] == grupo_cultivo) &
        (Inputs['ANIO'] == año) &
        (Inputs['NOMBRE_DEPARTAMENTO'] == departamento) &
        (Inputs['NOMBRE_MUNICIPIO'] == municipio) &
        (Inputs['NUM_CLUSTERS'] == valores_unicos)&
        (Inputs['NOMBRE_CULTIVO'] == cultivo)
    ]
    x_values = [1, 2, 3, 4, 5]
    y_values = [10, 8, 12, 6, 9]
    #x_values = filtered_data['FECHA']  # Reemplaza 'FECHA' con tu columna de fechas
    #y_values = filtered_data['RENDIMIENTO_TONELADAS_HA']  # Reemplaza con tus rendimientos

    fig = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode='lines', marker_color='#2cfec1'))
    #fig = px.line(data2, x='local_timestamp', y="Demanda total [MW]", markers=True, labels={"local_timestamp": "Fecha"})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                      plot_bgcolor='rgba(0,0,0,0)', 
                      font_color="#2cfec1",
                      font_size=14,
                      xaxis_title="Fecha",
                      yaxis_title="Predición de Cultivo")
    fig.update_xaxes(showgrid=True, gridwidth=0.25, gridcolor='#7C7C7C')
    fig.update_yaxes(showgrid=True, gridwidth=0.25, gridcolor='#7C7C7C')
    #fig.update_traces(line_color='#2cfec1')

    return fig



# Method to update prediction
@app.callback(
    Output(component_id='resultado', component_property='children'),
    [Input(component_id='dropdown-cultivo', component_property='value'), 
     Input(component_id='dropdown-año', component_property='value'), 
     Input(component_id='dropdown-numero-cluster', component_property='value')]
)
def update_output_div(cultivo, anio, cluster):
    print(f"cultivo: {cultivo}, anio: {anio}, cluster: {cluster}")
    myreq = {
        "inputs": [
            {
            "NOMBRE_CULTIVO": str(cultivo),
            "ANIO": anio,
            "NUM_CLUSTERS": cluster
            }
        ]
      }
    print(f"Contenido de myreq: {myreq}")
   
    headers =  {"Content-Type":"application/json", "accept": "application/json"}
    #print(f"Contenido de myreq: {headers}")
    # POST call to the API
    response = requests.post(api_url, data=json.dumps(myreq), headers=headers)
    data = response.json()
    logger.info("Response: {}".format(data))
    # Imprimir la respuesta completa del API
    print(f"Respuesta completa del API: {data}")

    # Pick result to return from json format
    result = data["predictions"][0]
    
    return result


if __name__ == '__main__':
    #logger.info("Running dash")
    app.run_server(debug=True,host= '0.0.0.0', port=8060)
