import ast
from typing import Any

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError

FORBIDDEN_PYTHON_MODULES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "importlib",
    "socket",
    "signal",
    "multiprocessing",
    "pathlib",
    "glob",
    "tempfile",
    "platform",
}

FORBIDDEN_PYTHON_FUNCTIONS = {
    "eval",
    "exec",
    "open",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "compile",
    "globals",
    "locals",
    "vars",
}

FORBIDDEN_SQL_QUERY = {
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
}


class SecurityVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            if alias.name in FORBIDDEN_PYTHON_MODULES:
                raise ValueError(f"Forbidden module import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if node.module in FORBIDDEN_PYTHON_FUNCTIONS:
            raise ValueError(f"Forbidden module import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        # Case 1: simple function name
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_PYTHON_FUNCTIONS:
                raise ValueError(f"Forbidden function call: {node.func.id}")

        # Case 2: attribute call (like math.sqrt)
        elif isinstance(node.func, ast.Attribute):
            full_name = (
                f"{node.func.value.id}.{node.func.attr}"
                if isinstance(node.func.value, ast.Name)
                else node.func.attr
            )
            if full_name in FORBIDDEN_PYTHON_FUNCTIONS:
                raise ValueError(f"Forbidden function call: {full_name}")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            raise ValueError(f"Forbidden attribute access: {node.attr}")
        self.generic_visit(node)


def validate_python(code: str) -> None:
    """Parse and validate python code string using SecurityVisitor."""
    tree = ast.parse(code)
    visitor = SecurityVisitor()
    visitor.visit(tree)


def validate_sql(query: str) -> None:
    """Parse and validate SQL query"""
    try:
        tree = sqlglot.parse_one(query, dialect="duckdb")
    except ParseError as e:
        raise SyntaxError(f"SQL syntax error: {e}") from e

    if tree.find(*FORBIDDEN_SQL_QUERY):
        raise ValueError("Only read-only SQL is allowed.")
