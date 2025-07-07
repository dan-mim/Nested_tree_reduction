from numpy import float64
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Union
from .FiltrationFuns import Atom

__all__ = ["ScenarioData", "ipms_socp", "non_anticipative_projection"]



@dataclass
class ScenarioData:
    # Scenario indice
    indice: int
    # Scenario value
    scenario: NDArray[float64]
    # Value of IPM's weight for solving the OCP with data from this class
    eps: Optional[float] = 0.
    # Authorize Warm-start for solving the OCP by IPMs. If True, the IPMs is solved using eps as initial weighting
    # parameter
    warm_start: Optional[bool] = False
    # Value of the optimal control, i.e. the solution of the RPHA for the scenario of this Scenario data
    control: Optional[NDArray[float64]] = None


def non_anticipative_projection(xs: list[NDArray[float64]],
                                filtration: list[list[Atom]],
                                probas: Optional[list[float]] = None) -> list[NDArray[float64]]:
    is_proba = probas is not None and len(probas) == len(xs)
    na_xs = xs.copy()
    for stage, stage_partition in enumerate(filtration):

        if len(stage_partition) >= len(xs):
            break

        for atom in stage_partition:
            if is_proba:
                mean_pbat_atom = sum(xs[scen_id][:, stage] * probas[scen_id] for scen_id in atom.scen_ids) / atom.proba
            else:
                mean_pbat_atom = sum(xs[scen_id][:, stage] for scen_id in atom.scen_ids) / len(atom.scen_ids)

            for scen_id in atom.scen_ids:
                na_xs[scen_id][:, stage] = mean_pbat_atom

    return na_xs

