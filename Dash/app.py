import dash
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import datetime as dt
import base64

app = dash.Dash(__name__)

Inputs = pd.read_csv('../data/Evaluaciones_Agropecuarias_Municipales_EVA.csv')

grupos_cultivos = Inputs['GRUPO \nDE CULTIVO'].unique().tolist()
años = Inputs['AÑO'].unique().tolist()
departamentos = Inputs['DEPARTAMENTO'].unique().tolist()
cultivos = Inputs['CULTIVO'].unique().tolist()

# Datos de ejemplo
data = {
    'Nombre': ['Juan', 'Ana', 'Luis', 'María'],
    'Edad': [25, 30, 35, 28],
    'Ciudad': ['Madrid', 'Barcelona', 'Sevilla', 'Valencia']
}
df = pd.DataFrame(data)

with open('img/Logo.png', 'rb') as f:
    logo_data = f.read()
encoded_logo = base64.b64encode(logo_data).decode()

app.layout = html.Div([
    html.Div([
        html.Img(src=f"data:image/png;base64,{encoded_logo}", height=100),
        html.H1("Rendimiento Agrícola por Hectárea"),
    ], className="header"),

     html.Div([
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
            html.Div("Card 1", className="card-title"),
            # Contenido de la Card 1
        ], className="card"),
        html.Div([
            html.Div("Card 2", className="card-title"),
            # Contenido de la Card 2
        ], className="card"),
        html.Div([
            html.Div("Card 3", className="card-title"),
            # Contenido de la Card 3
        ], className="card"),
        html.Div([
            dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
            # Contenido de la Card 6
        ], className="table-container"),  # Contenedor de la tabla
    ], className="cards-container"),  # Contenedor derecho
    
    html.Div([
            dcc.Graph(id="line-chart", className="line-chart"),
        ], className="model_graph"),
    ], className="main-container")
])



@app.callback(
    Output("dropdown-municipio", "options"),
    [Input("dropdown-departamento", "value")]
)
def update_municipios(departamento):
    # Filtra los municipios basados en el departamento seleccionado
    municipios = Inputs[Inputs['DEPARTAMENTO'] == departamento]['MUNICIPIO'].unique()
    options = [{'label': municipio, 'value': municipio} for municipio in municipios]
    return options

@app.callback(
    Output("line-chart", "figure"),
    [Input("dropdown-grupo-cultivo", "value"),
     Input("dropdown-año", "value"),
     Input("dropdown-municipio", "value"),
     Input("dropdown-departamento", "value"),
     Input("dropdown-cultivo", "value")]
)

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

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
