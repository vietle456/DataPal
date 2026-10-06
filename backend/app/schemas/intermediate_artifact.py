from pydantic import BaseModel


class DatasetArtifact(BaseModel):
    id: str
    storage_path: (
        str  # absolute host path (e.g. storage/intermediate/sql_results/result_xxx.parquet)
    )
    container_path: str  # container-side path (e.g. /workspace/intermediate/result_xxx.parquet)
