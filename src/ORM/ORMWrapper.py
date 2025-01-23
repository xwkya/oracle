"""
shared_orm.py

Generic ORM wrapper using SQLAlchemy. Handles common operations such as:
 - Creating tables for a given model
 - Dropping tables for a given model
 - Inserting and upserting records
 - Deleting records by primary key

All operations here are synchronous; for async needs, use run_in_executor or an async driver.
"""
import logging
from typing import Optional, Type, TypeVar, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from src.ORM.db_config import get_connection_string

Base = declarative_base()

ModelType = TypeVar("ModelType", bound=Base)

class ORMWrapper:
    def __init__(self, db_url: Optional[str] = None):
        """
        Initializes a new SharedORM instance.

        :param db_url: SQLAlchemy database URL.
        """
        connection_string = get_connection_string()
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(ORMWrapper.__name__)

    def create_table(self, model_class: Type[ModelType]):
        """
        Creates the table for the given model_class if it doesn't exist.
        """
        self.logger.debug(f"Creating table for class {model_class.__name__}")
        model_class.__table__.create(self.engine, checkfirst=True)

    def drop_table(self, model_class: Type[ModelType]):
        """
        Drops the table for the given model_class if it exists.
        """
        self.logger.debug(f"Dropping table for class {model_class.__name__}")
        model_class.__table__.drop(self.engine, checkfirst=True)

    def insert_record(self,
                      model_class: Type[ModelType],
                      **data) -> ModelType:
        """
        Inserts a single record into the table specified by model_class.
        Returns the newly inserted record.
        """
        session: Session = self.SessionLocal()
        self.logger.debug(f"Inserting record into {model_class.__name__}")

        try:
            record = model_class(**data)
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        finally:
            session.close()

    def upsert_record(self,
                      model_class: Type[ModelType],
                      pk_field: str,
                      pk_value: Any,
                      **data) -> ModelType:
        """
        Upserts a record in the table specified by model_class.

        :param model_class: The SQLAlchemy model class to upsert into.
        :param pk_field: The field name representing the primary key.
        :param pk_value: The primary key value to match or insert.
        :param data: Additional fields to set for the new or existing record.
        """
        session: Session = self.SessionLocal()
        self.logger.debug(f"Upserting record in {model_class.__name__} with {pk_field}={pk_value}")


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
