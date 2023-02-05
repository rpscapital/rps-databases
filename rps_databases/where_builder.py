import pandas as pd
from .operators import (
    Operator,
    Is
)

def build_where(**kwargs):

    conditions = []
    params = []

    for key, operator in kwargs.items():
        if not isinstance(operator, Operator):
            if isinstance(operator, pd.DataFrame):

                assert key in operator.columns, f"'{key}' não existe no DataFrame"
                operator = operator.get(key)

            operator = Is(operator).custom

        conditions.append(f"{key} {operator.build()}")
        params.extend(operator.params())

    where = " and ".join(conditions)

    if not len(where):
        return where, params

    where = "WHERE " + where

    return where, params