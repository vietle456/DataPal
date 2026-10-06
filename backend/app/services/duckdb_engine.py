import contextlib
import json
import re
from pathlib import Path

import duckdb
import sqlglot
from sqlglot import expressions as exp


class DuckDBEngine:
    """Service wrapper for DuckDB analytical database operations."""

    def __init__(self, db_path: str | Path | None = None, read_only: bool = False):
        # In-memory database or disk-persisted .duckdb file
        self.db_path = str(db_path) if db_path else ":memory:"
        self.conn = duckdb.connect(database=self.db_path, read_only=read_only)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        with contextlib.suppress(Exception):
            self.conn.close()

    def load_dataset(self, file_path: Path, table_name: str) -> bool:
        """Dynamically ingests CSV or Parquet into DuckDB table."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(
                f"Invalid table name '{table_name}': only letters, digits, "
                "and underscores are allowed, and it must start with a letter or underscore."
            )

        escaped_path = str(file_path.resolve()).replace("'", "''")

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            _NULL_STRINGS = [
                "N/A",
                "n/a",
                "NA",
                "na",
                "NULL",
                "null",
                "Null",
                "None",
                "none",
                "NONE",
                "NaN",
                "nan",
                "",
                "-",
                "?",
            ]
            null_list = "[" + ", ".join(f"'{s}'" for s in _NULL_STRINGS) + "]"
            query = f"CREATE OR REPLACE TABLE \"{table_name}\" AS SELECT * FROM read_csv('{escaped_path}', nullstr={null_list});"
        elif suffix == ".parquet":
            query = f"CREATE OR REPLACE TABLE \"{table_name}\" AS SELECT * FROM read_parquet('{escaped_path}');"
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        self.conn.execute(query)
        return True

    def get_schema_summary(self) -> str:
        """Returns table names, columns, data types, and sample rows for LLM context."""
        tables = self.conn.execute("SHOW TABLES;").fetchall()
        schema_info = {}

        for (table_name,) in tables:
            escaped_table = table_name.replace('"', '""')
            col_info = self.conn.execute(f"DESCRIBE {escaped_table};").fetchall()
            columns = [{"name": c[0], "type": c[1]} for c in col_info]

            # Fetch sample rows
            rows = self.conn.execute(f"SELECT * FROM {escaped_table} LIMIT 3").fetchall()
            samples = [dict(zip([c[0] for c in col_info], row, strict=False)) for row in rows]

            schema_info[table_name] = {"columns": columns, "sample_rows": samples}
        return json.dumps(schema_info, indent=2)

    def execute_read_query(self, query: str, max_rows: int = 500) -> dict:
        """Executes read-only SQL query with a safety LIMIT clause."""
        try:
            # Safety: force LIMIT 500 if not present
            tree = sqlglot.parse_one(query, dialect="duckdb")

            if not isinstance(tree, exp.Query):
                raise TypeError("Only query statements are allowed")

            if tree.args.get("limit") is None:
                tree = tree.limit(max_rows)

            query = tree.sql(dialect="duckdb")

            # Execute
            cursor = self.conn.execute(query)

            # column name + declared type from the cursor description
            columns = [{"name": desc[0], "type": str(desc[1])} for desc in cursor.description]
            col_names = [c["name"] for c in columns]

            # raw list-of-tuples
            rows = [dict(zip(col_names, row, strict=False)) for row in cursor.fetchall()]

            # lightweight numeric summary computed from raw values
            summary: dict = {"row_count": len(rows)}
            for col in columns:
                name = col["name"]
                values = [row[name] for row in rows if row[name] is not None]
                if values and all(isinstance(v, (int, float)) for v in values):
                    summary[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": round(sum(values) / len(values), 6),
                    }

            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "summary": summary,
            }
        except (sqlglot.errors.SqlglotError, TypeError, duckdb.Error) as e:
            return {"success": False, "error": str(e)}
