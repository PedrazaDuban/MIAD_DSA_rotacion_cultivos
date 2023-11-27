from typing import Any, List, Optional

from pydantic import BaseModel
from model.processing.validation import DataInputSchema

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]

# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "CODIGO_DEPARTAMENTO": 73,
                        "NOMBRE_DEPARTAMENTO": "TOLIMA",
                        "CODIGO_MUNICIPIO": 73226,
                        "NOMBRE_MUNICIPIO": "CUNDAY",
                        "GRUPO_CULTIVO": "FRUTALES",
                        "SUBGRUPO_CULTIVO": "MORA",
                        "NOMBRE_CULTIVO": "MORA",
                        "REGION_SISTEMA": "MORA", 
                        "ANIO": 2009,
                        "PERIODO": 2009,
                        "AREA_SIEMBRA_HA": 5.0,
                        "AREA_COSECHA_HA": 5,
                        "PRODUCCION_TONELADAS": 10,
                        "RENDIMIENTO_TONELADAS_HA": 2.0,
                        "ESTADO_PRODUCCION": "FRUTO FRESCO",
                        "NOMBRE_CIENTIFICO": "RUBUS GLAUCUS",
                        "CICLO_CULTIVO": "PERMANENTE",
                        "NUM_CLUSTERS": 3.0
                    }
                ]
            }
        }
