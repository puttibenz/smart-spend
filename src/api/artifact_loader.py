"""
SmartSpend AI - Dynamic Artifact Loader Module
Finds and loads the latest serialized model artifacts from models_artifacts/
without hardcoding timestamped filenames, enforcing fail-fast startup checks.
"""

import os
import glob
import json
import joblib
from typing import Dict, Any, Tuple

def load_latest_artifact(pattern: str, artifacts_dir: str = "models_artifacts") -> Tuple[Any, str]:
    """
    Finds the latest artifact matching the glob pattern and loads it using joblib.
    Raises FileNotFoundError if no matching artifact is found.
    """
    search_path = os.path.join(artifacts_dir, pattern)
    matching_files = glob.glob(search_path)
    
    if not matching_files:
        raise FileNotFoundError(
            f"[FATAL] Required model artifact matching pattern '{pattern}' was not found in '{artifacts_dir}'."
        )
        
    # Sort files lexicographically by timestamp in filename / mtime
    latest_file = sorted(matching_files, key=lambda f: (os.path.getmtime(f), f))[-1]
    
    try:
        artifact = joblib.load(latest_file)
        return artifact, latest_file
    except Exception as e:
        raise RuntimeError(f"[FATAL] Failed to load artifact from '{latest_file}': {str(e)}")

def load_all_phase1_phase2_artifacts(
    artifacts_dir: str = "models_artifacts",
    metrics_path: str = "outputs/metrics/phase1_metrics.json"
) -> Dict[str, Any]:
    """
    Dynamically discovers and loads:
    1. Phase 1 TF-IDF Vectorizer
    2. Phase 1 Best Categorization Classifier (determined from phase1_metrics.json)
    3. Phase 2 v2 ML Impulse Risk Model
    
    Fails fast at server startup if any component is missing.
    """
    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(f"[FATAL] Artifacts directory '{artifacts_dir}' does not exist.")
        
    # Determine best category model from phase1 metrics
    best_cat_model_name = "Logistic Regression"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                p1_metrics = json.load(f)
                best_cat_model_name = p1_metrics.get("best_ml_model", "Logistic Regression")
        except Exception:
            best_cat_model_name = "Logistic Regression"

    cat_model_pattern = "logreg_phase1_*.joblib" if "logistic" in best_cat_model_name.lower() else "lightgbm_phase1_*.joblib"

    # Load artifacts dynamically
    vec_obj, vec_path = load_latest_artifact("vectorizer_phase1_*.joblib", artifacts_dir)
    cat_obj, cat_path = load_latest_artifact(cat_model_pattern, artifacts_dir)
    impulse_obj, impulse_path = load_latest_artifact("impulse_model_phase2_*.joblib", artifacts_dir)
    
    # Handle wrapped artifact dict vs direct model object
    vec_model = vec_obj.get("vectorizer", vec_obj) if isinstance(vec_obj, dict) else vec_obj
    cat_model = cat_obj.get("model", cat_obj) if isinstance(cat_obj, dict) else cat_obj
    impulse_model = impulse_obj.get("model", impulse_obj) if isinstance(impulse_obj, dict) else impulse_obj
    
    return {
        "vectorizer": vec_model,
        "vectorizer_path": vec_path,
        "category_model": cat_model,
        "category_model_name": best_cat_model_name,
        "category_model_path": cat_path,
        "impulse_model_v2": impulse_model,
        "impulse_model_path": impulse_path,
        "impulse_artifact_meta": impulse_obj if isinstance(impulse_obj, dict) else {}
    }
