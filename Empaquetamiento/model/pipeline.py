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
                n_estimators=config.model_config.get('n_estimators', 100), 
                max_depth=config.model_config.get('max_depth', 3),
                min_samples_split=config.model_config.get('min_samples_split', 2),
                learning_rate=config.model_config.get('learning_rate', 0.1),
                random_state=config.model_config.get('random_state', None),
                loss=config.model_config.get('loss', 'ls'),
            ),
        ),
    ]
)
