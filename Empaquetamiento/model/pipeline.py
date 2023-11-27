from feature_engine.selection import DropFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, ensemble
from sklearn.preprocessing import LabelEncoder

from model.config.core import config
from model.processing import features as pp


cultivo_recomendado_pipe = Pipeline(
    [
        # Drop features 
        #("drop_features", 
        # DropFeatures(
        #     features_to_drop=[config.model_config.temp_features]
        #     )
        #),
        # Mappers
        #(
        #    "mapper_qual",
        #    pp.Mapper(
        #        variables=config.model_config.qual_vars,
        #        mappings=config.model_config.qual_mappings,
        #    ),
        #),
        # Scaler
        ("scaler", MinMaxScaler()
         ),
        # Random forest 
        ("GradientBoostingRegressor",
            ensemble.GradientBoostingRegressor(
                n_estimators = 500,
                max_depth = 4,                 
                min_samples_split = 5,
                learning_rate = 0.01,
                loss= 'squared_error',
            ),
        ),
    ]
)
