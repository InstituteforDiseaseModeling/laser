def get_storage_url():
    # MySQL Database Configuration (Docker Container)
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "root"  # noqa: S105
    MYSQL_HOST = "localhost"
    MYSQL_PORT = "3306"
    MYSQL_DB = "optuna"

    # MySQL Storage URL for Optuna
    storage_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

    return storage_url
