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
import geopandas as gpd
import matplotlib.pyplot as plt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app server
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

IP_API= '44.204.231.223'
# PREDICTION API URL 
api_url = os.getenv('API_URL')
api_url = "http://{}/api/v1/predict".format(api_url)


with open('../data/cultivos.csv', 'r', encoding='utf-8') as file:
    Inputs = pd.read_csv(file)

municipios_mapa=gpd.read_file('../data/MunicipiosVeredas19MB.json')

with open('../data/mapa_clusters.csv', 'r', encoding='utf-8') as file:
    mapa_data = pd.read_csv('../data/mapa_clusters.csv')


  

grupos_cultivos = Inputs['GRUPO_CULTIVO'].unique().tolist()
años = Inputs['ANIO'].unique().tolist()
departamentos = Inputs['NOMBRE_DEPARTAMENTO'].unique().tolist()
cultivos = Inputs['NOMBRE_CULTIVO'].unique().tolist()
Clusters = Inputs['NUM_CLUSTERS'].unique()

CantidadCultivos = len(Inputs['NOMBRE_CULTIVO'].unique())
NumClusters = len(Inputs['NUM_CLUSTERS'].unique())
valores_unicos = list(range(1, NumClusters + 1))
Toneladas = Inputs['RENDIMIENTO_TONELADAS_HA']


NumMunicipios = len(Inputs['NOMBRE_MUNICIPIO'].unique())
NumMunicipios_formateado = "{:,}".format(NumMunicipios)

##mapa
# Datos de ejemplo Tabla

df = pd.DataFrame(Inputs, columns=['NOMBRE_CULTIVO','NUM_CLUSTERS', 'RENDIMIENTO_TONELADAS_HA'])

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
html.Br(),  # Agregar un espacio vertical
html.Br(),  # Agregar un espacio vertical


html.Div([
        html.Button("Enviar Consulta",
                    id="enviar-consulta",
                    n_clicks=0,
                    style={'color': '#2cfec1', 'textAlign': 'center', 'fontSize': '18px'}),
        html.Div(id='output-message')
    ], style={'textAlign': 'center'}) 

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
        html.H4('Tabla deL TOP 15 de Recomendaciones', className="title-visualizacion"),
        dcc.Loading(
                    id="loading",
                    type="circle",
                    children=[
                        dash_table.DataTable(
                            id='recomendaciones-table',
                            columns=[{"name": i, "id": i} for i in df.columns],
                            style_cell={'backgroundColor': 'rgba(0,0,0,0)', 'color': '#2cfec1', 'textAlign': 'center'},
                        )
                    ]
                ),
    ], className="table-container"),  # Contenedor de la tabla
], className="cards-container"), # Contenedor de las cards y la tabla

# Contenedor de la visualización del mapa
html.Div([
    html.H4('Grupo de Cultivos con Municipicos Similares', className="title-visualizacion"),
    html.Img(src=f"data:image/png;base64,{encoded_mapa}"),
# Map plot
   # dcc.Graph(id='mapa'),
  
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
@app.callback(
    Output('output-message', 'children'),
    [Input('enviar-consulta', 'n_clicks')]
)
def boton(n_clicks):
    message = f"Se hizo clic en el botón {n_clicks} veces."
    return message



#Llamado a la base de card 1

#llamado a la base de tabla del dataset
@app.callback(
    Output('recomendaciones-table', 'data'),
    [
        Input('dropdown-departamento', 'value'),
        Input('dropdown-municipio', 'value'),
        Input("dropdown-grupo-cultivo", "value")
    ]
)
##################################Tabla#############################################################
#def update_table(departamento, municipio,grupos_cultivos):
#    # Filtra el DataFrame según los valores seleccionados en los dropdowns
#    filtered_df = Inputs
#    if departamento:
#        filtered_df = filtered_df[filtered_df['NOMBRE_DEPARTAMENTO'] == departamento]
#
#        if municipio:
#            filtered_df = filtered_df[filtered_df['NOMBRE_MUNICIPIO'] == municipio]
#
#            if grupos_cultivos:
#                filtered_df = filtered_df[filtered_df['GRUPO_CULTIVO'] == grupos_cultivos]
#
#        df_top10_sorted = filtered_df.sort_values(by='RENDIMIENTO_TONELADAS_HA', ascending=False)
#        df_top10_sorted_head10 = df_top10_sorted.head(15)
#    return df_top10_sorted_head10.to_dict('records')
#################################################################################################

def update_table(departamento, municipio,grupos_cultivos):
    # Filtra el DataFrame según los valores seleccionados en los dropdowns
    filtered_df = Inputs
    if departamento:
        filtered_df = filtered_df[filtered_df['NOMBRE_DEPARTAMENTO'] == departamento]

        if municipio:
            filtered_df = filtered_df[filtered_df['NOMBRE_MUNICIPIO'] == municipio]

            if grupos_cultivos:
                filtered_df = filtered_df[filtered_df['GRUPO_CULTIVO'] == grupos_cultivos]

        cluster_tabla=filtered_df['NUM_CLUSTERS'].iloc[0]
        tabla_filt= Inputs.loc[(Inputs['NUM_CLUSTERS'] == cluster_tabla) & (Inputs['GRUPO_CULTIVO'] == grupos_cultivos)]
        table_show=tabla_filt.pivot_table(index='NOMBRE_CULTIVO', aggfunc={'RENDIMIENTO_TONELADAS_HA':'mean', 'NUM_CLUSTERS':'mean'})
        table_show=table_show.reset_index()
        df_top10_sorted_head10 = table_show.nlargest(15, ['RENDIMIENTO_TONELADAS_HA'])

    return df_top10_sorted_head10.to_dict('records')

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
    # Filtra los datos basados en las selecciones
    filtered_data = Inputs[
        (Inputs['RENDIMIENTO_TONELADAS_HA'] == Toneladas) &
        (Inputs['ANIO'] == año)
    ]

    # Asigna los valores de x y y basados en los datos filtrados
    x_values = filtered_data['ANIO'].tolist()
    y_values = filtered_data['RENDIMIENTO_TONELADAS_HA'].tolist()
    #x_values = filtered_data['FECHA']  # Reemplaza 'FECHA' con tu columna de fechas
    #y_values = filtered_data['RENDIMIENTO_TONELADAS_HA']  # Reemplaza con tus rendimientos

    fig = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode='lines', marker_color='#2cfec1'))
    #fig = px.line(data2, x='local_timestamp', y="Demanda total [MW]", markers=True, labels={"local_timestamp": "Fecha"})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                      plot_bgcolor='rgba(0,0,0,0)', 
                      font_color="#2cfec1",
                      font_size=25,
                      xaxis_title="Fecha",
                      yaxis_title="Predición de Cultivo")
    fig.update_xaxes(showgrid=True, gridwidth=0.25, gridcolor='#7C7C7C')
    fig.update_yaxes(showgrid=True, gridwidth=0.25, gridcolor='#7C7C7C')
    #fig.update_traces(line_color='#2cfec1')

    return fig



# Method to update prediction
@app.callback(
    Output(component_id='resultado', component_property='children'),
    [[Input(component_id='enviar-consulta', component_property='n_clicks')],
     Input(component_id='dropdown-cultivo', component_property='value'), 
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

@app.callback(
    Output(component_id='resultado', component_property='children'),
    [[Input(component_id='enviar-consulta', component_property='n_clicks')],
     Input(component_id='dropdown-cultivo', component_property='value'), 
     Input(component_id='dropdown-año', component_property='value'), 
     Input(component_id='dropdown-numero-cluster', component_property='value')]
)
def update_output_div(n_clicks, cultivo, anio, cluster):
    if n_clicks > 0:
        myreq = {
            "inputs": [
                {
                    "NOMBRE_CULTIVO": str(cultivo),
                    "ANIO": anio,
                    "NUM_CLUSTERS": cluster
                }
            ]
        }
        headers = {"Content-Type": "application/json", "accept": "application/json"}
        response = requests.post(api_url, data=json.dumps(myreq), headers=headers)
        data = response.json()
        result = data["predictions"][0]
        return result
    else:
        return ""


@app.callback(
    dash.dependencies.Output('mapa', 'figure'),
    [dash.dependencies.Input('mapa', 'relayoutData')]
)
def update_map(relayoutData):
    # Aquí puedes agregar la lógica para actualizar el mapa dinámicamente
    # por ejemplo, puedes filtrar los datos según el área seleccionada en el mapa

    # Supongo que 'CODIGO_MUNICIPIO' está en tu DataFrame
    # Reemplaza esto con la columna correcta en tu DataFrame
    # y cualquier lógica de filtrado adicional que necesites

    filtered_data = mapa_data

    # Crea el mapa con Plotly Express
    fig = px.choropleth_mapbox(
        filtered_data,
        geojson=municipios_mapa,  # Reemplaza esto con tu archivo GeoJSON
        featureidkey='properties.CODIGO_MUNICIPIO',  # Reemplaza con tu clave
        locations='CODIGO_MUNICIPIO',  # Reemplaza con tu columna de ubicaciones
        color='tu_columna_de_color',  # Reemplaza con la columna que deseas mapear
        mapbox_style="carto-positron",
        center={"lat": 4.5709, "lon": -74.2973},  # Ajusta según tus necesidades
        zoom=5,
        opacity=0.5,
    )

    # Actualiza la figura del mapa
    return fig




if __name__ == '__main__':
    #logger.info("Running dash")
    app.run_server(debug=True,host= '0.0.0.0', port=8060)
