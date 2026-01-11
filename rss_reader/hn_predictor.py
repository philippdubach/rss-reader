"""HN Success Predictor using RoBERTa transformer model (V7).

Predicts the probability that a post title will be successful on Hacker News
(>100 points). Uses a fine-tuned RoBERTa model with isotonic calibration.

V7 improvements over V6:
- RoBERTa-only (dropped SBERT which added no value)
- Increased regularization (dropout 0.2, weight decay 0.05)
- Reduced overfitting (gap reduced from 0.109 to 0.042)
- Simpler deployment, faster inference
- AUC ~0.685 with better generalization
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class HNPredictor:
    """Predicts HN success probability for post titles.
    
    Lazily loads the model on first prediction to avoid startup overhead.
    Uses RoBERTa-base fine-tuned on HN data with isotonic calibration.
    """
    
    # Default model path relative to this file
    DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "hn_model_v7"
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the predictor.
        
        Args:
            model_path: Path to model directory. Uses bundled model if None.
        """
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self._model = None
        self._tokenizer = None
        self._calibrator = None
        self._config = None
        self._device = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy-load the model, tokenizer, and calibrator."""
        if self._loaded:
            return
        
        logger.info(f"Loading HN predictor from {self.model_path}")
        
        # Load config
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path) as f:
            self._config = json.load(f)
        
        # Determine device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        
        logger.info(f"Using device: {self._device}")
        
        # Load tokenizer and model
        roberta_path = self.model_path / "roberta"
        self._tokenizer = AutoTokenizer.from_pretrained(str(roberta_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(roberta_path))
        self._model.to(self._device)
        self._model.eval()
        
        # Load calibrator (isotonic or platt)
        isotonic_path = self.model_path / "isotonic_calibrator.joblib"
        platt_path = self.model_path / "platt_calibrator.joblib"
        
        if isotonic_path.exists():
            self._calibrator = joblib.load(isotonic_path)
            self._calibrator_type = "isotonic"
            logger.info("Loaded isotonic calibrator")
        elif platt_path.exists():
            self._calibrator = joblib.load(platt_path)
            self._calibrator_type = "platt"
            logger.info("Loaded Platt calibrator")
        else:
            self._calibrator_type = None
            logger.warning("No calibrator found, using raw probabilities")
        
        self._loaded = True
        logger.info(f"HN predictor loaded (v{self._config.get('version', 'unknown')})")
    
    @property
    def threshold(self) -> float:
        """Get the optimal classification threshold."""
        self._load_model()
        return self._config.get("optimal_threshold", 0.5)
    
    @property
    def metrics(self) -> dict:
        """Get model performance metrics."""
        self._load_model()
        return self._config.get("metrics", {})
    
    def predict_batch(self, titles: list[str], batch_size: int = 32) -> list[float]:
        """Predict HN success probability for a batch of titles.
        
        Args:
            titles: List of post titles to score
            batch_size: Batch size for inference
            
        Returns:
            List of probabilities (0.0 to 1.0) for each title
        """
        if not titles:
            return []
        
        self._load_model()
        
        all_probs = []
        
        # Process in batches
        for i in range(0, len(titles), batch_size):
            batch_titles = titles[i:i + batch_size]
            
            # Tokenize
            inputs = self._tokenizer(
                batch_titles,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use softmax for 2-class output
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            
            # Apply calibration if available
            if self._calibrator is not None:
                if self._calibrator_type == "platt":
                    # Platt uses predict_proba
                    probs = self._calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
                else:
                    # Isotonic uses predict
                    probs = self._calibrator.predict(probs)
            
            all_probs.extend(probs.tolist())
        
        return all_probs
    
    def predict(self, title: str) -> float:
        """Predict HN success probability for a single title.
        
        Args:
            title: Post title to score
            
        Returns:
            Probability (0.0 to 1.0)
        """
        return self.predict_batch([title])[0]
    
    def predict_with_recommendation(self, title: str) -> dict:
        """Predict with full recommendation details.
        
        Args:
            title: Post title to score
            
        Returns:
            Dict with probability, is_hit prediction, and confidence level
        """
        prob = self.predict(title)
        threshold = self.threshold
        
        # Confidence levels based on distance from threshold
        if prob >= 0.7:
            confidence = "high"
        elif prob >= threshold:
            confidence = "medium"
        elif prob >= 0.2:
            confidence = "low"
        else:
            confidence = "very_low"
        
        return {
            "probability": prob,
            "is_hit": prob >= threshold,
            "confidence": confidence,
            "threshold": threshold,
        }


# Singleton instance for efficiency
_predictor_instance: Optional[HNPredictor] = None


def get_predictor() -> HNPredictor:
    """Get the singleton HNPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = HNPredictor()
    return _predictor_instance
