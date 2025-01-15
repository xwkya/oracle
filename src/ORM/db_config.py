import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_connection_string():
    return (
        f"mssql+pyodbc://"
        f"{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_SERVER')}:{os.getenv('DB_PORT')}/"
        f"{os.getenv('DB_NAME')}"
        "?driver=ODBC+Driver+18+for+SQL+Server"
        "&TrustServerCertificate=no"
        "&Encrypt=yes"
    )