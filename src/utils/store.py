import os
import pandas as pd
import pickle
import json
from typing import Any, Dict

import functools

from src.models.classifier import Classifier

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data/raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data/processed")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")



class InvalidExtension(Exception):
    pass



def _check_filepath(ext):
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            filepath = kwargs.get("filepath")
            if not filepath:
                filepath = args[1]
            
            if not filepath.endswith(ext):
                raise InvalidExtension(f"{filepath} has invalid extension, require {ext}")
            
            return f(*args, **kwargs)
        
        return _wrapper
    return _decorator




class Store:
    @_check_filepath(".csv")
    def get_csv(self, filepath : str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(filepath, **kwargs)

    @_check_filepath(".csv")
    def put_csv(self, filepath: str, df: pd.DataFrame ,**kwargs) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"File must be a pandas DataFrame, recieved a {type(df)}")
        df.to_csv(filepath, index = False, **kwargs)

    @_check_filepath(".pkl")
    def get_pkl(self, filepath: str) -> Any:
        with open(filepath, 'rb') as f: 
            return pickle.load(f)

    @_check_filepath(".pkl")
    def put_pkl(self, filepath: str, python_object: Any) -> None:
        if not python_object:
            raise TypeError("Python object must be non-zero, non-empty and not none")
        with open(filepath, "wb") as f:
            pickle.dump(python_object, f)

    @_check_filepath(".json")
    def get_json(self, filepath: str) -> Dict:
        with open(filepath, "r") as f:
            return json.load(f)

    @_check_filepath(".json")    
    def put_json(self, filepath, dic: Dict) -> None:
        if not isinstance(dic, Dict):
            raise TypeError(f"dic must be of type Dict, got {type(dic)}")
        with open(filepath, "w") as f:
            json.dump(dic, f)


class AssignmentStore(Store):
    project_dir = PROJECT_DIR
    raw_data_dir = RAW_DATA_DIR
    processed_data_dir = PROCESSED_DATA_DIR
    model_dir = MODEL_DIR
    output_dir = OUTPUT_DIR

    def get_raw(self, filepath : str, **kwargs) -> pd.DataFrame:
        filepath = os.path.join(self.raw_data_dir, filepath)
        return self.get_csv(filepath, **kwargs)
    
    def get_processed(self, filepath : str, **kwargs) -> pd.DataFrame:
        filepath = os.path.join(self.processed_data_dir, filepath)
        return self.get_csv(filepath, **kwargs)

    def put_processed(self, filepath : str, df: pd.DataFrame ,**kwargs) -> None:
        filepath = os.path.join(self.processed_data_dir, filepath)
        self.put_csv(filepath, df, **kwargs)

    def get_model(self, filepath : str) -> Classifier:
        filepath = os.path.join(self.model_dir, filepath)
        return self.get_pkl(filepath)

    def put_model(self, filepath : str, model : Classifier) -> None:
        filepath = os.path.join(self.model_dir, filepath)
        self.put_pkl(filepath, model)

    def get_metrics(self, filepath : str, **kwargs) -> Dict['str' , 'float']:
        filepath = os.path.join(self.output_dir, filepath)
        return self.get_json_(filepath)

    def put_metrics(self, filepath : str, metrics: Dict['str' , 'float']) -> None:
        filepath = os.path.join(self.output_dir, filepath)
        self.put_json(filepath, metrics)

    def get_predictions(self, filepath : str, **kwargs) -> pd.DataFrame:
        filepath = os.path.join(self.output_dir, filepath)
        return self.get_csv(filepath)

    def put_predictions(self, filepath : str, df: pd.DataFrame, **kwargs) -> None:
        filepath = os.path.join(self.output_dir, filepath)
        self.put_csv(filepath, df)