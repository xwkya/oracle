"""
shared_orm.py

Generic ORM wrapper using SQLAlchemy. Handles common operations such as:
 - Creating tables for a given model
 - Dropping tables for a given model
 - Inserting and upserting records
 - Deleting records by primary key

All operations here are synchronous; for async needs, use run_in_executor or an async driver.
"""

from typing import Optional, Type, TypeVar, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base

Base = declarative_base()

ModelType = TypeVar("ModelType", bound=Base)

class SharedORM:
    def __init__(self, db_url: Optional[str] = None):
        """
        Initializes a new SharedORM instance.

        :param db_url: SQLAlchemy database URL. Defaults to a local SQLite file if None.
        """
        if db_url is None:
            db_url = "sqlite:///news_summaries.db"
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_table(self, model_class: Type[ModelType]):
        """
        Creates the table for the given model_class if it doesn't exist.
        """
        model_class.__table__.create(self.engine, checkfirst=True)

    def drop_table(self, model_class: Type[ModelType]):
        """
        Drops the table for the given model_class if it exists.
        """
        model_class.__table__.drop(self.engine, checkfirst=True)

    def insert_record(self, model_class: Type[ModelType], **data) -> ModelType:
        """
        Inserts a single record into the table specified by model_class.
        Returns the newly inserted record.
        """
        session: Session = self.SessionLocal()
        try:
            record = model_class(**data)
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        finally:
            session.close()

    def upsert_record(self, model_class: Type[ModelType], pk_field: str, pk_value: Any, **data) -> ModelType:
        """
        Upserts a record in the table specified by model_class.

        :param pk_field: The field name representing the primary key.
        :param pk_value: The primary key value to match or insert.
        :param data: Additional fields to set for the new or existing record.
        """
        session: Session = self.SessionLocal()
        try:
            existing = session.query(model_class).filter(getattr(model_class, pk_field) == pk_value).first()
            if existing:
                for k, v in data.items():
                    setattr(existing, k, v)
                session.commit()
                session.refresh(existing)
                return existing
            else:
                record = model_class(**data)
                session.add(record)
                session.commit()
                session.refresh(record)
                return record
        finally:
            session.close()

    def delete_record(self, model_class: Type[ModelType], pk_field: str, pk_value: Any):
        """
        Deletes a record matching pk_field == pk_value from model_class if it exists.
        """
        session: Session = self.SessionLocal()
        try:
            record = session.query(model_class).filter(getattr(model_class, pk_field) == pk_value).first()
            if record:
                session.delete(record)
                session.commit()
        finally:
            session.close()
