from enum import Enum
from dataclasses import dataclass, field
from typing import List
import numpy as np

class Sense(Enum):
    MAX = 1
    MIN = -1

class Constraint(Enum):
    LEQ = "<="
    GEQ = ">="
    EQ = "="

@dataclass
class Problem:
    sense: Sense
    c: np.ndarray
    A: List[np.ndarray] = field(default_factory=list)
    b: List[float] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)

    def add_constraint(self, coefficients: list, constraint_type: Constraint, rhs: float):
        self.A.append(np.array(coefficients, dtype=float))
        self.constraints.append(constraint_type)
        self.b.append(float(rhs))