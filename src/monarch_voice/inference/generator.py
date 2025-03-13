# Copyright (c) Monarch Voice-1 (MV1) Project
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from fairseq2.generation import Sampling


@dataclass
class SequenceGeneratorOptions:
    """Options to control sequence generation for Monarch Voice models."""

    beam_size: int = 1
    """The size of beam search."""

    soft_max_seq_len: Tuple[float, int] = (1.0, 200)
    """The maximum sequence length as a tuple of (coefficient, constant)."""

    len_penalty: float = 1.0
    """The length penalty coefficient."""

    logits_processor: Optional[Callable[[Any], Any]] = None
    """Optional logits processor to manipulate the logits before sampling/beam search."""

    top_k: Optional[int] = None
    """The k value for top-k sampling."""

    top_p: Optional[float] = None
    """The p value for nucleus sampling."""

    temperature: Optional[float] = None
    """The temperature for sampling."""

    sampling_strategy: Optional[Sampling] = None
    """The sampling strategy to use.""" 