from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List

from . import models

class Regression_Data(BaseModel):
    X: List[float]
    y: List[float]

class DatasetPostRequest(BaseModel):
    name: str
    problem_type: str
    data: Regression_Data

def add_new_dataset(
    db: Session, 
    dataset: DatasetPostRequest
):
    """Adds a new dataset to the database."""
    existing_dataset = retrieve_dataset(db, dataset.name)
    if existing_dataset:
        return existing_dataset
    
    db_dataset = models.Dataset_Storage(
        dataset_name = dataset.name,
        problem_type = dataset.problem_type,
        data = dataset.data.model_dump_json()
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

def retrieve_dataset(
    db: Session,
    dataset_name: str
):
    """Retrieves one dataset with specified name."""
    return (db.query(models.Dataset_Storage)
            .filter(models.Dataset_Storage.dataset_name == dataset_name)
            .first())

def retrieve_dataset_names(
    db: Session,
):
    """Retrieves every dataset name currently in storage."""
    return (db.query(models.Dataset_Storage.dataset_name)
            .all())
    
def remove_dataset(
    db: Session,
    dataset_name: str
):
    """Removes the dataset with specified name"""
    dataset = retrieve_dataset(db, dataset_name)
    if dataset:
        db.delete(dataset)
        db.commit()
    
    return dataset

def clear_all(
    db: Session
):
    """
    Removes all entries in the database
    """
    dataset_entries = retrieve_dataset_names(db)
    names = [row.dataset_name for row in dataset_entries]
    
    for name in names:
        remove_dataset(db, name)
    
    return dataset_entries