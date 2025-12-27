from datetime import datetime

from pydantic import BaseModel
from typing import Dict, List, Optional, Any


class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]]
    model_name: str
    stage: Optional[str] = "Production"


class PredictionResponse(BaseModel):
    predictions: List[float]


class IcebergQueryRequest(BaseModel):
    filter_condition: Optional[str] = None
    limit: Optional[int] = 100


class TimeTravelRequest(BaseModel):
    snapshot_id: Optional[int] = None
    timestamp: Optional[str] = None


class MLPipelineRunRequest(BaseModel):
    train_only: bool = False
    skip_validation: bool = False
    resume_from_checkpoint: bool = False


class MLPipelineRunResponse(BaseModel):
    success: bool
    run_id: Optional[str]
    metrics: Dict[str, Any] = {}
    pipeline_metrics: Dict[str, Any]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    error: Optional[str] = None


class IngestionRunRequest(BaseModel):
    run_adverse_media: bool = True
    optimize_table: bool = False


class IngestionRunResponse(BaseModel):
    success: bool
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: Optional[float]

    table_statistics: Optional[Dict[str, Any]]
    adverse_media_analyzed: Optional[bool]
    optimize_table: Optional[bool]

    counts: Optional[Dict[str, Any]]
    pipeline_metrics: Optional[Dict[str, Any]]

    error: Optional[str] = None