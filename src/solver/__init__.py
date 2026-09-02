"""by lyuwenyu
"""

from .solver import BaseSolver
from .det_solver_enhanced import DetSolverEnhanced

from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolverEnhanced,
}