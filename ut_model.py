import json
import os
from datetime import datetime
from typing import Any, Literal

import joblib
import sklearn
from joblib import dump
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

joblib_version = joblib.__version__


class ModelDetails(BaseModel):
    """모델의 기본 정보를 담는 스키마."""
    type: Literal["ml", "dl", "genai"]
    algorithm: str
    library: str
    library_version: str
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)
    joblib_version: str


class DataSchema(BaseModel):
    """모델 학습에 사용된 피처 및 타겟 변수 스키마."""
    features: list[str]
    target: str


class ModelFiles(BaseModel):
    """저장된 모델 및 메타데이터 파일 경로."""
    model_path: str
    metadata_path: str


class ModelInfo(BaseModel):
    """모델 메타데이터의 최종 구조."""
    model_details: ModelDetails
    data_schema: DataSchema
    equation: str  # VIF/p-value 튜닝 과정에서 생성된 최종 OLS 방정식을 저장
    ols_summary_text: str # OLS 최종 summary() 텍스트 저장
    hyperparameters: dict[str, Any]
    model_files: ModelFiles


def save_model_and_metadata(model: Pipeline, params: dict, output_path: str, feature_names: list, target_name: str, equation_str: str, ols_summary: str):
    """
    scikit-learn 파이프라인 모델과 메타데이터(json)를 저장합니다.
    ols_summary: statsmodels의 최종 summary().as_text() 결과.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. 모델 저장
    dump(model, output_path)
    print(f"Model saved: {output_path}")

    json_path = os.path.splitext(output_path)[0] + ".json"
    
    # 2. 모델 알고리즘 이름 파악
    algorithm_name = "OLS Linear Regression (sklearn Pipeline)"


    # 3. ModelInfo 객체 생성
    model_info = ModelInfo(
        model_details=ModelDetails(
            type="ml",
            algorithm=algorithm_name,
            library="scikit-learn",
            library_version=sklearn.__version__,
            joblib_version=joblib_version,
        ),
        data_schema=DataSchema(features=feature_names, target=target_name),
        equation=equation_str,  # 외부에서 계산된 최종 방정식을 사용
        ols_summary_text=ols_summary, # statsmodels 최종 요약 텍스트
        hyperparameters=params,
        model_files=ModelFiles(model_path=os.path.basename(output_path), metadata_path=os.path.basename(json_path)),
    )

    # 4. 메타데이터 저장
    with open(json_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False 옵션을 사용하여 한글이 깨지지 않도록 합니다.
        f.write(model_info.model_dump_json(indent=4, exclude_none=True))
    print(f"Meta saved: {json_path}")


def load_model_info_from_json(json_path: str) -> ModelInfo:
    """JSON 파일에서 모델 메타데이터를 불러옵니다."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Meta not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ModelInfo.model_validate(data)