from sqlalchemy import create_engine, Column, String, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///database.db', echo=True)
Base = declarative_base()

class Dataset_Storage(Base):
    __tablename__ = 'datasets_storage'

    id = Column(Integer, primary_key=True)
    dataset_name = Column(String, unique=True, nullable=False)
    problem_type = Column(String, nullable=False)
    data = Column(JSON)

Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()