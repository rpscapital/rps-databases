import mysql.connector
import sqlalchemy
import os
import psycopg2
import pandas as pd
import numpy as np
import urllib.parse
import typing
import sys
import typing
from . import common
from .where_builder import build_where

def build_agg(**kwargs):
    ...


class Table():
    def __init__(self, db, schema, name: str):
        self.schema = schema
        self.name = name
        self.db = db

    def path(self):
        return f'{self.schema.name}.{self.name}'

    def get(self,
        *args,
        distinct: bool = False,
        **kwargs
    ):

        columns = list(args)

        if isinstance(columns, list):
            columns = ", ".join(columns)

        if isinstance(columns, str):
            columns = columns

        if distinct:
            columns = "DISTINCT " + columns

        return self.db.select(columns=columns, origin=self.path(), **kwargs)
class Schema():
    def __init__(self, db, name: str):
        self.db = db
        self.name = name

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return Table(db=self.db, schema=self, name=attr)

class Database():

    def __init__(self, engine):
        self.engine = engine
        self.connect()

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return Schema(db=self, name=attr)

    def __sanitize_params(self, query, params):
        if params is None:
            return None

        if isinstance(params, list):
            params = tuple(params)

        if not isinstance(params, tuple):
            params = (params,)

        if not len(params):
            return None

        params = tuple([
            tuple(x) if
            common.is_iterable(x)
            else x
            for x in params
        ])

        return params

    def connect(self):
        self.engine.connect()
        self.con = self.engine.raw_connection()
        self.cur = self.con.cursor()

    def commit(self):
        self.con.commit()

    def mogrify(self, query: str, params=None):
        return self.cur.mogrify(query, params).decode("utf-8")

    def execute(self, query: str, params=None):
        return self.cur.execute(query, params)

    def fetch(self, sql, params: tuple = None):
        """
            db.fetch(""\"--sql
                SELECT *
                FROM my.table
                WHERE mycolumn = %s
            ""\", params=[value])
        """

        params = self.__sanitize_params(sql, params)

        if not hasattr(self.cur, 'mogrify'):
            return pd.read_sql(sql, self.engine, params=params)

        sql = self.cur.mogrify(sql.strip(), params).decode().replace("%", "%%")

        return pd.read_sql(sql, self.engine)

    def select(self,
               columns: typing.Union[str, list],
               origin: str,
               groupby: typing.Union[str, list] = "",
               **kwargs):

        if isinstance(columns, list):
            columns = ", ".join(columns)

        if not len(groupby) and ('(' in columns or ')' in columns):
            e_columns = enumerate(columns.split(", "))

            groupby = [str(i + 1) for i, c in e_columns if '(' not in c or ')' not in c]

        if isinstance(groupby, list):
            groupby = ", ".join(groupby)

        where, params = build_where(**kwargs)

        if isinstance(groupby, str):
            if len(groupby):
                groupby = f"GROUP BY {groupby}"


        if not len(columns.strip()):
            columns = "*"

        SQL = f"""
            SELECT
            {columns}
            FROM {origin}
            {where}
            {groupby}
        """

        return self.fetch(SQL, params)



    def insert(self, df, schema: str, table: str, commit=True):
        return self.execute_values(df, schema, table, commit)

    def upsert(self, df, schema: str, table: str, on_conflict: list, update: list, page_size=5000, commit=True):
        return self.execute_batch(df, schema, table, on_conflict=(on_conflict, update), page_size=page_size, commit=commit)

    def update(self, df: pd.DataFrame, schema: str, table: str, primary_key: list, columns: list, commit=True):

        SQL = """--sql
            UPDATE {0}.{1} {2}
            SET {3}
            FROM (VALUES {4}) AS df ({5})
            WHERE {6}
        """
        alias = f"{schema[0]}{table[0]}"
        values = [tuple(x) for x in df[[*primary_key, *columns]].replace({np.nan:None}).to_numpy()]

        SQL = SQL.format(
            schema, table, alias,
            ", ".join([f"{x} = df.{x}" for x in columns]),
            ", ".join(["%s"] * len(values)), ", ".join([*primary_key, *columns]),
            " AND ".join([f"{alias}.{x} = df.{x}" for x in primary_key])
        )

        SQL = self.cur.mogrify(SQL, tuple(values)).decode()
        self.cur.execute(SQL)

        if not commit: return

        self.commit()

    def execute_values(self, df, schema: str, table: str, commit=True):
        """
        Using psycopg2.extras.execute_values() to insert the dataframe
        """
        # Create a list of tupples from the dataframe values
        nan = {np.nan:None}
        df = df.astype(object).replace(nan).replace(nan)

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

        if not commit: return

        self.commit()

    def execute_batch(self, df, schema: str, table: str, page_size=5000, commit=True, on_conflict: tuple = None):
        """
        Using psycopg2.extras.execute_batch() to insert the dataframe
        """
        # Create a list of tupples from the dataframe values
        nan = {np.nan:None}
        df = df.astype(object).replace(nan).replace(nan)


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

        if not commit: return

        self.commit()

    def register(self, data: pd.DataFrame, schema: str, table: str, index: str, commit=True):
        sql = f'SELECT * FROM {schema}.{table}'
        registered_data = pd.read_sql(sql, self.engine, index_col=index)

        data = data[~data[index].isin(
            registered_data.index)]
        if len(data) > 0:
            self.execute_batch(data, schema, table, commit=commit)


def create_engine(engine: str, host: str, user: str, password: str, database: str, app_name: str = None):
    if not host:
        raise Exception('Informe um host')

    if not user:
        raise Exception('Informe um user')

    if not password:
        raise Exception('Informe um password')

    application_name = ''
    if engine == 'postgresql+psycopg2':
        application_name = f'?application_name={app_name or sys.argv[0][-16:] or "Default"}'

    password = urllib.parse.quote_plus(password)

    engine = sqlalchemy.create_engine(
        f'{engine}://{user}:{password}@{host}/{database}{application_name}', echo=False)

    return Database(engine)

def connect(database: str = "rps", host: str = "", user: str = "", password: str = "", dialect: str = 'postgresql', app_name: str = None) -> Database:

    dialects = {
        'postgresql': 'postgresql+psycopg2',
        'mysql': 'mysql'
    }

    assert dialect in dialects.keys(), 'Available "dialects": %s' % list(dialects.keys())

    host = os.getenv(f"{dialect.upper()}_HOST", host)
    user = os.getenv(f"{dialect.upper()}_USER", user)
    password = os.getenv(f"{dialect.upper()}_PASS", password)
    app_name = app_name or os.getenv('APP_NAME')

    return create_engine(dialects[dialect], host, user, password, database, app_name)

def cloud_connect(database: str = "rps", host: str = "", user: str = "", password: str = "", app_name: str = None) -> Database:

    host = os.getenv("POSTGRESQL_HOST", host)
    user = os.getenv("POSTGRESQL_USER", user)
    password = os.getenv("POSTGRESQL_PASS", password)
    app_name = app_name or os.getenv('APP_NAME')

    return create_engine('postgresql+psycopg2', host, user, password, database, app_name)


def local_connect(database: str = "", host: str = "", user: str = "", password: str = "") -> Database:

    host = os.getenv("MYSQL_HOST", host)
    user = os.getenv("MYSQL_USER", user)
    password = os.getenv("MYSQL_PASS", password)

    return create_engine('mysql', host, user, password, database)