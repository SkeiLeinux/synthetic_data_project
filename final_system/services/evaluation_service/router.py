# services/evaluation_service/router.py

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # -> final_system/
from evaluator.privacy.privacy_evaluator import PrivacyConfig, PrivacyEvaluator
from evaluator.utility.utility_evaluator import UtilityConfig, UtilityEvaluator
from shared.schemas.evaluation import PrivacyEvalRequest, UtilityEvalRequest
from services.evaluation_service.settings import Settings, get_settings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # already set above, idempotent
from shared.log_context import set_run_id  # noqa: E402

router = APIRouter()
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"{label} не найден: {path}"},
        )
    return pd.read_csv(path)


def _resolve_synth(settings: Settings, synth_path: str) -> Path:
    """synth_path — путь относительно data_root или абсолютный."""
    p = Path(synth_path)
    if p.is_absolute():
        return p
    return settings.data_root / p


# ── POST /evaluate/privacy ────────────────────────────────────────────────────

@router.post(
    "/evaluate/privacy",
    status_code=status.HTTP_200_OK,
    summary="Оценка приватности синтетических данных",
)
def evaluate_privacy(
    body: PrivacyEvalRequest,
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    set_run_id(body.run_id)
    t0 = time.time()
    logger.info("Privacy eval started: split_id=%s synth_path=%s", body.split_id, body.synth_path)
    split_dir = settings.splits_dir / body.split_id
    real_train = _load_csv(split_dir / "train.csv", "train.csv")
    real_holdout = _load_csv(split_dir / "holdout.csv", "holdout.csv")
    synth = _load_csv(_resolve_synth(settings, body.synth_path), "synthetic.csv")
    logger.info("Privacy eval: train=%d holdout=%d synth=%d", len(real_train), len(real_holdout), len(synth))

    config = PrivacyConfig(
        quasi_identifiers=body.quasi_identifiers,
        sensitive_attribute=body.sensitive_attribute,
        compute_classical=bool(body.quasi_identifiers and body.sensitive_attribute),
    )
    evaluator = PrivacyEvaluator(config)
    result = evaluator.evaluate(real_train, real_holdout, synth, dp_report=body.dp_report)
    logger.info("Privacy eval done in %.1fs", time.time() - t0)
    return result


# ── POST /evaluate/utility ────────────────────────────────────────────────────

@router.post(
    "/evaluate/utility",
    status_code=status.HTTP_200_OK,
    summary="Оценка полезности синтетических данных",
)
def evaluate_utility(
    body: UtilityEvalRequest,
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    set_run_id(body.run_id)
    t0 = time.time()
    logger.info("Utility eval started: split_id=%s target=%s", body.split_id, body.target_column)
    split_dir = settings.splits_dir / body.split_id
    real_train = _load_csv(split_dir / "train.csv", "train.csv")
    real_holdout = _load_csv(split_dir / "holdout.csv", "holdout.csv")
    synth = _load_csv(_resolve_synth(settings, body.synth_path), "synthetic.csv")

    config = UtilityConfig(target_column=body.target_column)
    evaluator = UtilityEvaluator(config)
    # real_test_df = holdout: отложенная выборка, которую генератор не видел
    result = evaluator.evaluate(real_train, synth, real_holdout)
    logger.info("Utility eval done in %.1fs", time.time() - t0)
    return result
