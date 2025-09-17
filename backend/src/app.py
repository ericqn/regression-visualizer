from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.databases.operations import (
    DatasetPostRequest, 
    add_new_dataset, 
    retrieve_dataset_names, 
    retrieve_dataset,
    remove_dataset,
    clear_all
)
from src.databases.models import get_db
import json

class NameRequest(BaseModel):
    datset_name: str 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


@app.delete('/remove-specific-dataset')
async def remove_specific_dataset(name: str, db: Session = Depends(get_db)):
    return remove_dataset(db, dataset_name=name)


@app.post("/add-dataset")
def add_to_database(dataset: DatasetPostRequest, db: Session = Depends(get_db)):
    try:
        return add_new_dataset(db, dataset)
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))


@app.get("/get-dataset-names")
async def get_dataset_names(db: Session = Depends(get_db)):
    dataset_names = [row.dataset_name for row in retrieve_dataset_names(db)]

    return {'names': dataset_names}


@app.get("/get-specific-dataset")
async def get_dataset(name: str, db: Session = Depends(get_db)):
    dataset_info = retrieve_dataset(db, dataset_name=name)
    return {
        'dataset_name': dataset_info.dataset_name,
        'problem_type': dataset_info.problem_type,
        'data': json.loads(dataset_info.data)
    }


@app.delete('/debug-clear-all')
async def debug_clear_all(db: Session = Depends(get_db)):
    deleted_entries = [row.dataset_name for row in clear_all(db)]

    return {'deleted_entries': deleted_entries}


@app.post("/debug")
async def post_debug_endpoint(request: Request):
    data = await request.json()
    return {"sent_data": data}