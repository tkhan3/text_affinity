from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from text_affinity.controller import *
import os
import json
from enum import Enum
from typing import Optional,List,Dict
from compute.computation_engine import computation_engine


os.environ['LOAD_SELECTED_MODELS'] = 'tfidf,use,wmd_gensim'

app = FastAPI()
general_config = read_general_config()
storage_path,stopwords,logger = setup(general_config)
logger.info("Sample Stop Words Are {}".format(stopwords[1:10]))
loaded_models = load_models(general_config,logger)


input_data_ex = {"req_id":"xxxx","source" :["Obama speaks to media in Illinois","President greets the press in Chicago"],
              "target":["President greets the press in Chicago"],
              #"algorithms" :[{"name":"wmd_gensim","preprocess":True,"hyper_parameters":[]}], #{"name":"wmd_gensim", "preprocess":True}],
			  "algorithms" :[{"name":"tfidf","preprocess":True,"hyper_parameters":[],"preprocess_steps":['convert_lower']},
			                 {"name":"wmd_gensim","preprocess":True,"hyper_parameters":[],"preprocess_steps":[]},
			  				{"name":"spacy","preprocess":True,"hyper_parameters":[],"preprocess_steps":[]}],
              "custom_stopwords" : ["tanveer"]}

class algorithmsDetails(BaseModel):
    name: str
    preprocess: Optional[bool] = True
    hyper_parameters: Optional[list] = None
    preprocess_steps: Optional[list] = None


class requestAffinity(BaseModel):
    req_id: str
    source: list
    target: list
    algorithms: List[algorithmsDetails]
    custom_stopwords: list

    class Config:
        schema_extra = {"example": input_data_ex}

class modelNames(str,Enum):
    sentence_transformer = "sentence_transformer"
    spacy = "spacy"
    universal_sentence_encoder = "universal_sentence_encoder"
    glove = "glove"
    fasttext = "fasttext"
    wmd_gensim = "wmd_gensim"
    wmd_wml = "wmd_wml"
    tfidf = "tfidf"

class affinityResponse(BaseModel):
    status_code: int
    message: str
    req_id: str
    affinity: List

@app.get("/status_check")
def read_status():
    return {"Welcome tO Text Affinity Server is Up and Running"}

@app.get("/query_loaded_models")
def find_loaded_models():
    name_loaded_models = list(loaded_models.keys())
    return {"message": "loaded models are","loaded_models": name_loaded_models}

@app.get("/query_loaded_models/{model_name}")
def find_selected_model(model_name: modelNames):
    if model_name.value in list(loaded_models.keys()):
        return {"message": "requested model is loaded","model_name":model_name.value}
    else:
        return {"message": "requested model is not loaded","model_name":model_name.value}

@app.post("/compute_text_affinity/",response_model=affinityResponse)
def compute_text_affinity(input_data_obj: requestAffinity):
    input_data = json.loads(input_data_obj.json())
    affinity_list = computation_engine(input_data, loaded_models, stopwords, storage_path, general_config, logger)
    logger.info(affinity_list)
    response_obj = {}
    response_obj['status_code'] = 200
    response_obj['message'] = "success"
    response_obj['affinity'] = affinity_list
    response_obj['req_id'] = input_data['req_id']
    logger.info(response_obj)
    return response_obj