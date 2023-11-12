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


app = dash.Dash(__name__)

Inputs = pd.read_csv('../data/Evaluaciones_Agropecuarias_Municipales_EVA.csv')

grupos_cultivos = Inputs['GRUPO \nDE CULTIVO'].unique().tolist()
años = Inputs['AÑO'].unique().tolist()
departamentos = Inputs['DEPARTAMENTO'].unique().tolist()
cultivos = Inputs['CULTIVO'].unique().tolist()


total_departamentos = len(Inputs['DEPARTAMENTO'].unique())
# Datos de ejemplo
data = {
    'Nombre': ['Juan', 'Ana', 'Luis', 'María', 'Pedro'],
    'Edad': [25, 30, 35, 28, 42],
    'Ciudad': ['Madrid', 'Barcelona', 'Sevilla', 'Valencia', 'Zaragoza']
}
df = pd.DataFrame(data)

with open('img/Logo.png', 'rb') as f:
    logo_data = f.read()
encoded_logo = base64.b64encode(logo_data).decode()

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
html.H6("Año"),
dcc.Dropdown(
id="dropdown-año",
options=[{'label': str(año), 'value': año} for año in años],
value=años[0],
searchable=True  # Habilitar la opción de búsqueda
),
html.H6("Grupo de Cultivo"),
dcc.Dropdown(
id="dropdown-grupo-cultivo",
options=[{'label': grupo, 'value': grupo} for grupo in grupos_cultivos],
value=grupos_cultivos[0],
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
)

], className="left-container"),  # Contenedor izquierdo

html.Div([
html.Div([             
# Contenedor cards            

html.Div([             
# Contenedor cards
html.Div([                  
    html.Div("Card 1", className="card-title"),
    html.Div(id="card-Valor"),
], className="card"),           


html.Div([
html.Div("Card 2", className="card-titdle"),
# Contenido de la Card 2
], className="card"),
html.Div([
html.Div("Card 3", className="card-title"),
# Contenido de la Card 3
], className="card"),# Contenedor cards
# Contenedor de la tabla
html.Div([
html.H4('Tabla de Recomendaciones', className="title-visualizacion"),
dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]), 
], className="table-container"),  # Contenedor de la tabla
], className="cards-container"), # Contenedor de las cards y la tabla


# Contenedor de la visualización del mapa
html.Div([
html.H4('Grupo de Cultivos con Municipicos Similares', className="title-visualizacion"),
dcc.RadioItems(
id='candidate',
options=["Joly", "Coderre", "Bergeron"],
value="Coderre",
inline=True
),
dcc.Graph(id="mapa", className="mapa-graph"), 

],className="mapa-container"),
# Contenedor de la tabla

html.Div([
html.H4('Rendimiento en Toneladas por Hectarea', className="title-line-chart"),
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
    municipios = Inputs[Inputs['DEPARTAMENTO'] == departamento]['MUNICIPIO'].unique()
    options = [{'label': municipio, 'value': municipio} for municipio in municipios]
    return options


#Llamado a la base de card 1

@app.callback(
    Output("card-Valor", "children"),
    Input("trigger-button", "n_clicks"),  # Un botón que dispara la actualización
)
def update_card(total_departamentos):
    # Supongamos que tienes una variable total_departamentos previamente calculada
    return f"Total de Departamentos: {total_departamentos}"


##mapa #https://plotly.com/python/maps/

@app.callback(
    Output("mapa", "figure"), 
    Input("candidate", "value"))


def display_choropleth(candidate):
    df = px.data.election() # replace with your own data source
    geojson = px.data.election_geojson()
    map = px.choropleth(
        df, geojson=geojson, color=candidate,
        locations="district", featureidkey="properties.district",
        projection="mercator", range_color=[0, 6500])
    map.update_geos(fitbounds="locations", visible=False)
    map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    map.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#2cfec1")
    return map 



##line chart
@app.callback(
    Output("line-chart", "figure"),
    [Input("dropdown-grupo-cultivo", "value"),
     Input("dropdown-año", "value"),
     Input("dropdown-municipio", "value"),
     Input("dropdown-departamento", "value"),
     Input("dropdown-cultivo", "value")]
)


#
def update_line_chart(grupo_cultivo, año, municipio, departamento, cultivo):
    # Filtra los datos basados en las selecciones
    filtered_data = Inputs[
        (Inputs['GRUPO \nDE CULTIVO'] == grupo_cultivo) &
        (Inputs['AÑO'] == año) &
        (Inputs['DEPARTAMENTO'] == departamento) &
        (Inputs['MUNICIPIO'] == municipio) &
        (Inputs['CULTIVO'] == cultivo)
    ]
    x_values = [1, 2, 3, 4, 5]
    y_values = [10, 8, 12, 6, 9]
    #x_values = filtered_data['FECHA']  # Reemplaza 'FECHA' con tu columna de fechas
    #y_values = filtered_data['RENDIMIENTO_TONELADAS_HA']  # Reemplaza con tus rendimientos

    fig = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode='lines'))
    #fig = px.line(data2, x='local_timestamp', y="Demanda total [MW]", markers=True, labels={"local_timestamp": "Fecha"})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#2cfec1")
    fig.update_xaxes(showgrid=True, gridwidth=0.25, gridcolor='#7C7C7C')
    fig.update_yaxes(showgrid=True, gridwidth=0.25, gridcolor='#7C7C7C')
    #fig.update_traces(line_color='#2cfec1')

    return fig
if __name__ == '__main__':
    app.run_server(debug=True)
