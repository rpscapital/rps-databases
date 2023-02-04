from . import common

class Operator:
    def __init__(self, value, operator):
        self.value = value
        self.operator = operator

    def build(self):
        return f"{self.operator} %s"

    def params(self):
        return [self.value]

class Equals(Operator):
    def __init__(self, value):
        super().__init__(value, "=")

class NotEquals(Operator):
    def __init__(self, value):
        super().__init__(value, "<>")

class GreaterThan(Operator):
    def __init__(self, value):
        super().__init__(value, ">")

class GreaterOrEqualsThan(Operator):
    def __init__(self, value):
        super().__init__(value, ">=")

class LessThan(Operator):
    def __init__(self, value):
        super().__init__(value, "<")

class LessOrEqualsThan(Operator):
    def __init__(self, value):
        super().__init__(value, "<=")

class IsIn(Operator):
    def __init__(self, value):
        super().__init__(value, "in")

class NotIn(Operator):
    def __init__(self, value):
        super().__init__(value, "not in")

class Like(Operator):
    def __init__(self, value):
        super().__init__(value, "like")

class Ilike(Operator):
    def __init__(self, value):
        super().__init__(value, "ilike")

class Between(Operator):
    def __init__(self, value):
        self.value = value
        self.operator = "between"

    def build(self):
        return f"{self.operator} %s and %s"

    def params(self):
        return self.value

class Not(Operator):
    def __init__(self, value):
        if common.is_iterable(value):
            self.not_class = NotIn(value)
        else:
            self.not_class = NotEquals(value)

    def build(self):
        return self.not_class.build()

    def params(self):
        return self.not_class.params()

class Is(Operator):
    def __init__(self, value):
        if common.is_iterable(value):
            self.not_class = IsIn(value)
        else:
            self.not_class = Equals(value)

    def build(self):
        return self.not_class.build()

    def params(self):
        return self.not_class.params()

class IsNull(Operator):
    def __init__(self):
        self.operator = "is null"

    def build(self):
        return f"{self.operator}"

    def params(self):
        return []

class NotNull(Operator):
    def __init__(self):
        self.operator = "notnull"

    def build(self):
        return f"{self.operator}"

    def params(self):
        return []