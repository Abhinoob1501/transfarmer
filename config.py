
from pathlib import Path

def get_config():
    return{
        "batch_size": 32,
        "num_epochs": 10,
        "lr":10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang1": "en",
        "lang2": "it",
        "model_folder":"weights",
        "model_basename":"tmodel_",
        "preload":None,
        "tockenizer_file":"tockenizer_{0}.json",
        "experiment_name":"run/tmodel"        
    }
def get_weight_file_path(config,epoch:str):
    model_folder=config["model_folder"]
    model_basename=config["model_basename"]
    model_filename=f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)
