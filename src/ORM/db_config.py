import os
import struct
from urllib import parse
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from sqlalchemy import event

# Load environment variables
load_dotenv()

# Constants for Azure SQL authentication
TOKEN_URL = "https://database.windows.net/"
SQL_COPT_SS_ACCESS_TOKEN = 1256


def setup_azure_token_provider(engine):
    """
    Sets up the Azure token provider for SQL Server authentication.
    This needs to be called before any connections are made.
    """
    azure_credentials = DefaultAzureCredential()

    @event.listens_for(engine, "do_connect")
    def provide_token(dialect, conn_rec, cargs, cparams):
        # Remove the "Trusted_Connection" parameter that SQLAlchemy adds
        cargs[0] = cargs[0].replace(";Trusted_Connection=Yes", "")

        # Create token credential
        raw_token = azure_credentials.get_token(TOKEN_URL).token.encode("utf-16-le")
        token_struct = struct.pack(f"<I{len(raw_token)}s", len(raw_token), raw_token)

        # Apply it to keyword arguments
        cparams["attrs_before"] = {SQL_COPT_SS_ACCESS_TOKEN: token_struct}


def get_connection_string() -> str:
    """
    Creates the connection string for Azure SQL Database.
    The string is intentionally minimal since the token provider will handle authentication.
    """
    return (
        f"mssql+pyodbc://@{os.getenv('DB_SERVER')}:{os.getenv('DB_PORT')}/"
        f"{os.getenv('DB_NAME')}?driver=ODBC+Driver+18+for+SQL+Server"
    )