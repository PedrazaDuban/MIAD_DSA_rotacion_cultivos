import numpy as np
from config.core import config
from pipeline import cultivo_recomendado_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def run_training() -> None:
    """Train the model."""
    
    # read training data
    data = load_dataset(file_name=config.app_config.train_data_file)
    # Create arrary of categorial variables to be encoded
    categorical_cols = ['NOMBRE_CULTIVO']
    le = LabelEncoder()
    # apply label encoder on categorical feature columns
    data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
    data[['ANIO','NOMBRE_CULTIVO', 'NUM_CLUSTERS']],  # predictors
    data['RENDIMIENTO_TONELADAS_HA'],
    test_size=config.model_config['test_size'],
    random_state=config.model_config['random_state'],
)
    #y_train = np.log(y_train)
    y_train = y_train.map(config.model_config.qual_mappings)

    # fit model
    cultivo_recomendado_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=cultivo_recomendado_pipe)


if __name__ == "__main__":
    run_training()
