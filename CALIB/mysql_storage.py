import os


def get_storage_url():
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "root"  # noqa: S105
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")  # Use localhost if outside Docker
    MYSQL_PORT = "3306"
    MYSQL_DB = "optuna_db"

    return f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
