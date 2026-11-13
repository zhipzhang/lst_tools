from .base import LSTDL2Table
from .observed import LSTObservedDL2Table
from .simulated import LSTSimulatedDL2Table

LSTDL2TableBase = LSTDL2Table
LSTDL2EventTable = LSTObservedDL2Table
LSTDL2MCTable = LSTSimulatedDL2Table

__all__ = [
    "LSTDL2EventTable",
    "LSTDL2MCTable",
    "LSTDL2Table",
    "LSTDL2TableBase",
    "LSTObservedDL2Table",
    "LSTSimulatedDL2Table",
]
