import mysql.connector
import sqlalchemy
from sqlalchemy import event
import os
import psycopg2
import pandas as pd
import dotenv

dotenv.load_dotenv()

class Database():
    def __init__(self, engine):
        self.engine = engine
        self.connect()

    def connect(self):
        self.engine.connect()
        self.con = self.engine.raw_connection()
        self.cur = self.con.cursor()

    def execute_values(self, df, schema: str, table: str, commit=True):
        """
        Using psycopg2.extras.execute_values() to insert the dataframe
        """
        # Create a list of tupples from the dataframe values
        df = df.where(pd.notnull(df), None)

        if not len(df):
            return

        tuples = [tuple(x) for x in df.to_numpy()]
        # Comma-separated dataframe columns
        cols = ','.join(list(df.columns))
        vals = ",".join(["%s" for x in df.columns])
        # SQL quert to execute
        query = f'INSERT INTO {schema}.{table}({cols}) VALUES '
        query += ",".join([self.cur.mogrify(f'({vals})', x).decode('utf-8')
                           for x in tuples])
        self.cur.execute(query)

        if commit:
            self.commit()

    def execute_batch(self, df, schema: str, table: str, page_size=5000, commit=True, on_conflict: tuple = None):
        """
        Using psycopg2.extras.execute_batch() to insert the dataframe
        """
        # Create a list of tupples from the dataframe values
        df = df.where(pd.notnull(df), None)

        if not len(df):
            return

        tuples = [tuple(x) for x in df.to_numpy()]
        # Comma-separated dataframe columns
        cols = ','.join(list(df.columns))
        vals = ",".join(["%s" for x in df.columns])
        # SQL quert to execute
        query = f'INSERT INTO {schema}.{table}({cols}) VALUES({vals})'

        if on_conflict:
            keys, values = on_conflict
            if not isinstance(keys, list):
                keys = [keys]
            if not isinstance(values, list):
                values = [values]
            keys = ",".join(keys)

            values = [f'{x} = excluded.{x}' for x in values]
            values = ",".join(values)
            query += f' ON CONFLICT ({keys}) DO UPDATE SET {values}'

        psycopg2.extras.execute_batch(self.cur, query, tuples, page_size)

        if commit:
            self.commit()

    def commit(self):
        self.con.commit()

    def register(self, data: pd.DataFrame, schema: str, table: str, index: str, commit=True):
        sql = f'SELECT * FROM {schema}.{table}'
        registered_data = pd.read_sql(sql, self.engine, index_col=index)

        data = data[~data[index].isin(
            registered_data.index)]
        if len(data) > 0:
            self.execute_batch(data, schema, table, commit=commit)


def create_engine(engine: str, host: str, user: str, password: str, database: str):
    if not host:
        raise Exception('Informe um host')

    if not user:
        raise Exception('Informe um user')
    
    if not password:
        raise Exception('Informe um password')
        
    engine = sqlalchemy.create_engine(
        f'{engine}://{user}:{password}@{host}/{database}', echo=False)

    return Database(engine)


def cloud_connect(database: str = "rps", host: str = "", user: str = "", password: str = "") -> Database:

    host = host or os.getenv("POSTGRESQL_HOST")
    user = user or os.getenv("POSTGRESQL_USER")
    password = password or os.getenv("POSTGRESQL_PASS")

    return create_engine('postgresql+psycopg2', host, user, password, database)


def local_connect(database: str = "", host: str = "", user: str = "", password: str = "") -> Database:

    host = host or os.getenv("MYSQL_HOST")
    user = user or os.getenv("MYSQL_USER")
    password = password or os.getenv("MYSQL_PASS")

    return create_engine('mysql', host, user, password, database)
