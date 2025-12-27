from fastapi import APIRouter, Depends

from src.server.deps import (
    get_mlflow_manager,
    get_iceberg_manager,
    get_config,
    get_ml_pipeline, get_ingestion_pipeline
)
from src.server.schema import (
    PredictionResponse,
    PredictionRequest,
    IcebergQueryRequest,
    TimeTravelRequest, MLPipelineRunRequest, MLPipelineRunResponse, IngestionRunResponse, IngestionRunRequest,
)
from src.server.service import ModelService, IcebergService, IngestionService
from src.utils.helper import create_spark_session

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(
    req: PredictionRequest,
    mlflow_manager=Depends(get_mlflow_manager),
    ml_pipeline=Depends(get_ml_pipeline)
):
    spark = create_spark_session(get_config())
    service = ModelService(spark, mlflow_manager, ml_pipeline)

    predictions = service.predict(
        model_name=req.model_name, records=req.records, stage=req.stage
    )
    return {"predictions": predictions}


@router.post("/iceberg/query")
def query_iceberg(
    req: IcebergQueryRequest, iceberg_manager=Depends(get_iceberg_manager)
):
    service = IcebergService(iceberg_manager)
    return service.query(req.filter_condition, req.limit)


@router.get("/iceberg/stats")
def iceberg_stats(iceberg_manager=Depends(get_iceberg_manager)):
    return iceberg_manager.get_table_statistics()


@router.post("/iceberg/time-travel")
def iceberg_time_travel(
    req: TimeTravelRequest, iceberg_manager=Depends(get_iceberg_manager)
):
    service = IcebergService(iceberg_manager)
    return service.time_travel(req.snapshot_id, req.timestamp)


@router.post("/pipeline/ml/run", response_model=MLPipelineRunResponse)
def run_ml_pipeline(
    req: MLPipelineRunRequest,
    mlflow_manager=Depends(get_mlflow_manager),
    ml_pipeline=Depends(get_ml_pipeline)
):
    spark = create_spark_session(get_config())
    service = ModelService(spark, mlflow_manager, ml_pipeline)

    result = service.run_pipeline(
        train_only=req.train_only,
        skip_validation=req.skip_validation,
        resume_from_checkpoint=req.resume_from_checkpoint
    )
    return result


@router.post("/pipeline/data/run", response_model=IngestionRunResponse)
def run_data_pipeline(
    req: IngestionRunRequest,
    ingestion_pipeline=Depends(get_ingestion_pipeline)
):
    service = IngestionService(ingestion_pipeline)

    result = service.run_ingestion(
        run_adverse_media=req.run_adverse_media,
        optimize_table=req.optimize_table
    )
    return result