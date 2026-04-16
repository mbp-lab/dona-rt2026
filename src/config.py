from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

class ConversationGapFilteringConfig(BaseModel):
    top_n: Optional[int] = None
    gap_fraction: Optional[float] = None
    rt_max_gap_filter: Optional[int] = None

class AggregationConfig(BaseModel):
    message_aggregation: Literal["message", "conversation", "donor"] = Field(default="message")
    aggregation_type: Optional[Literal["mean", "median", "probability"]] = Field(default="median")
    aggregation_period: Optional[int] = None

    @model_validator(mode="after")
    def check_aggregation_period_required(self):
        if self.aggregation_type == "probability" and self.aggregation_period is None:
            raise ValueError(
                "aggregation_period is required when aggregation_type is 'probability'."
            )
        return self

class ExperimentConfig(BaseModel):
    block_merge: Literal["first", "last"] = "first"
    rt_threshold: Optional[int] = None
    wc_threshold: Optional[Tuple[Literal["impute_median", "remove"], int]] = None
    conversation_min_wc_threshold: Optional[int] = None
    conversation_min_msg_threshold: Optional[int] = None
    exclude_donations_lt_n_chats: Optional[int] = None
    rt_minimum_threshold: Optional[int] = None
    conversation_gap: Optional[ConversationGapFilteringConfig] = ConversationGapFilteringConfig()
    system_message_filtering_threshold: Optional[int] = None
    jsd_base: Optional[float] = 2 # Base for the logarithm when calculating JSD
    lmm_formula: Optional[str] = None
    n_boot: int = Field(ge=1, default=500)
    predictors: Optional[List[str]] = None

    # Whether to round values up to (t % 60 == 0) without treating t as minutes
    round_to_minute: Optional[bool] = None

    # Whether to (t /= 60) and treat everything as minutes
    temporal_resolution: Optional[Literal["minute", "second"]] = Field(default="second")
    filter_temporal_resolution: Optional[Literal["seconds_only"]] = Field(default=None)
    model: Optional[Literal["Lm", "Lmer", "GLMM"]] = Field(default="Lmer")
    experiment_name: Optional[str] = None
    optimizer: Literal["bobyqa", "Nelder_Mead", "nloptwrap", "optimx"] = Field(default="bobyqa")
    optCtrl: Optional[Dict[str, Any]] = None
    aggregation: Optional[AggregationConfig] = AggregationConfig()
    distribution: str = Field(default="gaussian")

    def __repr__(self):
        return f'{self.__repr_name__()}(\n  {self.__repr_str__(",\n  ")}\n)'

def _load_yaml(path: Path) -> Dict:
    """Load a YAML file from the given path and return it as dictionary.

    Args:
        path (Path): Location of the file.

    Returns:
        Dict: The loaded YAML file.
    """
    with open(path, "r") as file:
        return dict(yaml.safe_load(file))

def load_experiment_config(
    path: Union[Path, str] = "config.yml",
) -> ExperimentConfig:
    """Load a YAML configuration file from a given path.

    Args:
        path (Path, optional): Location of the configuration file.
            Defaults to config.yml.

    Returns:
        ExperimentConfig: The loaded configuration.
    """
    return ExperimentConfig(**_load_yaml(path))