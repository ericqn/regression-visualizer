from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import pytest

from src.databases.operations import (
    Regression_Data, 
    DatasetRequest,
    add_new_dataset,
    retrieve_dataset,
    remove_dataset,
)
from src.databases.models import Dataset_Storage, Base

test_engine = create_engine("sqlite:///:memory:", echo=False)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope='function')
def db_session():
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_add_many_datasets(db_session):
    N_DATASETS = 1000
    for idx in range(N_DATASETS):
        dummy_data = Regression_Data(
            X = [1.0, 2.0],
            y = [2.0, 4.0],
            noise_values = [-0.25, 0.62],
            noise_strength = 1.0
        )
        dataset = DatasetRequest(
            name=f'Test dataset {idx}',
            problem_type='quadratic',
            data=dummy_data
        )
        add_new_dataset(db_session, dataset=dataset)
    
    assert retrieve_dataset(db_session, dataset_name=f'Test dataset {N_DATASETS-1}')
    assert retrieve_dataset(db_session, dataset_name='Test dataset 0')
    assert not retrieve_dataset(db_session, dataset_name='OOmpaloompa')


def test_add_invalid_data(db_session):
    with pytest.raises(ValueError):
        dummy_data = Regression_Data(
            X = ['str'],
            y = [2.0, 3.0, 4.0],
            noise_values = [-0.25, 0.5, 0.25],
            noise_strength = 1.0
        )

    with pytest.raises(ValueError):
        dummy_data = Regression_Data(
            X = [1.0,2.0,3.0],
            y = [2.0, 3.0, 4.0],
            noise_values = [-0.25, 0.5, 0.25],
        )


def test_prevent_duplicate_set_name(db_session):
    dummy_data = Regression_Data(
        X = [1.0, 2.0],
        y = [2.0, 4.0],
        noise_values = [-0.25, 0.62],
        noise_strength = 1.0
    )
    dataset = DatasetRequest(
        name=f'Duplicate dataset',
        problem_type='quadratic',
        data=dummy_data
    )
    add_new_dataset(db_session, dataset=dataset)

    Regression_Data(
        X = [3.0, 7.0],
        y = [1.25, -0.234],
        noise_values = [0.92, 0.32],
        noise_strength = 0.85
    )
    duplicate_dataset_name = DatasetRequest(
        name=f'Duplicate dataset',
        problem_type='quadratic',
        data=dummy_data
    )
    with pytest.raises(IntegrityError):
        add_new_dataset(db_session, dataset=dataset)


def test_remove_dataset(db_session):
    N_DATASETS = 3
    for idx in range(N_DATASETS):
        data = Regression_Data(
            X = [1.0,2.0,3.0],
            y = [23.0, 24.0, 25.0],
            noise_values=[0.5,-0.5,0.25],
            noise_strength=1.2
        )

        test_dataset = DatasetRequest(
            name = f'Test Dataset {idx}',
            problem_type='log',
            data = data
        )

        add_new_dataset(db_session, test_dataset)
    
    remove_dataset(db_session, 'Test Dataset 1')
    assert not remove_dataset(db_session, 'Test Dataset 1')
    assert not retrieve_dataset(db_session, 'Test Dataset 1')
