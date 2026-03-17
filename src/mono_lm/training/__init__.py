"""Character-level mono-lm training workflow."""

from .config import TrainingConfig, load_training_config
from .corpus import PreparedCorpus, load_prepared_corpus, prepare_corpus
from .vocabulary import CharacterVocabulary

__all__ = [
    "CharacterVocabulary",
    "PreparedCorpus",
    "TrainingConfig",
    "load_prepared_corpus",
    "load_training_config",
    "prepare_corpus",
]

try:
    from .generation import generate_from_checkpoint, latest_checkpoint_path
    from .trainer import TrainingResult, train_model

    __all__.extend(
        [
            "TrainingResult",
            "generate_from_checkpoint",
            "latest_checkpoint_path",
            "train_model",
        ]
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
