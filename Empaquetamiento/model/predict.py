import typing as t

import numpy as np
import pandas as pd

from model import __version__ as _version
from model.config.core import config
from model.processing.data_manager import load_pipeline
from model.processing.validation import validate_inputs
from sklearn.preprocessing import LabelEncoder

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_cultivo_recomendado_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        # Create arrary of categorial variables to be encoded
        categorical_cols = ['NOMBRE_CULTIVO']
        le = LabelEncoder()
        # apply label encoder on categorical feature columns
        validated_data[categorical_cols] = validated_data[categorical_cols].apply(lambda col: le.fit_transform(col))
        
        predictions = _cultivo_recomendado_pipe.predict(
            X=validated_data[['ANIO','NOMBRE_CULTIVO', 'NUM_CLUSTERS']]
        )
        results = {
            "predictions": [pred for pred in predictions], 
            "version": _version,
            "errors": errors,
        }

    return results
