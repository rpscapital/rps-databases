from .operators import (
    Operator,
    Is
)

def build_where(**kwargs):
    conditions = []
    params = []
    for key, value in kwargs.items():
        if not isinstance(value, Operator):
            value = Is(value)

        conditions.append(f"{key} {value.build()}")
        params.extend(value.params())
    where = " and ".join(conditions)

    if not len(where):
        return where, params

    where = "WHERE " + where
    return where, params