from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SubScore:
    name: str
    top_level: str
    options: List[str]


class Score:
    def __init__(self, scores: List[SubScore]):
        self._scores: Dict[str, SubScore] = {sc.top_level: sc for sc in scores}

    def get_score(
        self, feature_hash, answer_hash
    ) -> Optional[Tuple[SubScore, int]]:
        subscore = self._scores.get(feature_hash)

        if subscore is None:
            return None

        if answer_hash in subscore.options:
            return subscore, subscore.options.index(answer_hash)
        else:
            return None

    def __call__(
        self, feature_hash: str, answer_hash: str
    ) -> Optional[Tuple[SubScore, int]]:
        return self.get_score(feature_hash, answer_hash)
