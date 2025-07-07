import numpy as np
from collections import deque
from numba.typed import List, Dict
from numba import int32, float32, optional, jit, njit, types
from numba.experimental import jitclass
import numba as nb


class Atom:
    def __init__(self, elements: set, depth: int, n_id=0):
        self._atom = elements
        self._parent = None
        self._children = None
        self._depth = depth
        self._value = np.zeros((1,))
        self._proba = 0.
        self.id = n_id

    def depth(self):
        return self._depth

    def value(self):
        return self._value

    def add(self, x):
        self._atom.add(x)

    def pick(self,):
        return next(iter(self._atom))

    def __iter__(self,):
        return self._atom.__iter__()

    def elements(self,):
        return self._atom

    def is_empty(self):
        return len(self._atom) == 0

    def set_parent(self, parent_atom):
        self._parent = parent_atom

    def parent(self,):
        return self._parent

    def set_children(self, children: list):
        self._children = children

    def children(self,):
        return self._children

    def size(self,):
        return len(self._atom)

    def set_value(self, x):
        self._value = x

    def set_proba(self, x):
        self._proba = x

    def proba(self):
        return self._proba


def compute_filtration(
        predicted_scenarios: list, tol=1e-6, probas=[], check_proba=True
):
    """
    :param predicted_scenarios: List of scenarios, each scenario is a n x Times array where n is the dimension of the random process
    :param quantiles: List of quantiles. The list has n elements n being the dimension of the random process
    :param tol: tolerance for clustering
    :param method: either "quantile" or "distance"
    :return: filtration as a list of list of atoms
    """
    # Test if the list probas is indeed a probability
    if len(probas) == len(predicted_scenarios):
        if check_proba:
            assert (np.abs(sum(probas) - 1.) <= 1e-8 and min(probas) >= -1e-8), ("Scenarios do not form a probabilty: probas = " + str(probas))
    elif len(probas) == 0:
        inv_nscen = 1. / len(predicted_scenarios)
        probas = [inv_nscen for _ in range(len(predicted_scenarios))]
    else:
        raise Exception("Probas length does not match number of scenarios")

    # Definition of the set of scenario's number
    elements = set()
    # Initialize root value to a zero-nd.array of size equal to the number of random variable
    val_root = np.zeros((predicted_scenarios[0].shape[0],))

    # Root's elements is the set containing all scenario numbers
    # The root value is the expectation of the random variables at initial step
    root_id = 0
    for i in range(len(predicted_scenarios)):
        elements.add(i)
        val_root += predicted_scenarios[i][:, 0] * probas[i]

    # Build root Atom
    root_atom = Atom(elements, 0, root_id)
    root_atom.set_value(val_root)
    root_atom.set_proba(sum(probas))

    # Initialize the filtration as a list whose first element is a list containing only the root Atom
    filtration = [[root_atom]]

    # Traversing of the time-steps to compute the t-Partitions of the set of scenarios
    for t_ind in range(predicted_scenarios[0].shape[1] - 1):

        # Initialize the list of Atom at stage t_ind corresponding to the (t_ind+1)-partition of the set of scenario
        atoms_at_next_step = []

        # Now let us go through all atoms at this stage and compute its _children, i.e. the scenarios from this atom
        # that become distinct at next time-step.
        for atom in filtration[t_ind]:

            children_atoms = _compute_this_atom_children(t_ind, predicted_scenarios, atom, tol, probas)

            # link the _children to the parent atom
            atom.set_children(children_atoms)
            atoms_at_next_step += children_atoms

        # Add t-partition represented by a list of atom to the t-partition
        filtration.append(atoms_at_next_step)
    return filtration


def _compute_this_atom_children(
        t_ind: int, predicted_scenarios: list[np.ndarray], this_atom: Atom, tol: float, probas: list) -> list[Atom]:
    """
    :param t_ind:  Current time indice
    :param predicted_scenarios: 2D array whose rows are the scenarios for the stochastic process
    :param this_atom: Atom at t_ind such that all scenarios in this atom match the actual realization up to t_ind
    :param tol: Error tolerance on relative distances of scenarios values
    :param probas: List of probability of each scenario
    :return: this_atom _children
    """
    # This atom's _children list instanciation
    children_atoms = []

    # Initialization of the set of remaining scenarios as the set of elements in current atom
    remaining_scenarios = this_atom.elements().copy()

    atom_id = this_atom.id + 1
    # While loop to scatter the scenarios in the atom's _children
    while len(remaining_scenarios) > 0:
        # Get a scenario from remaining set
        this_scen = next(iter(remaining_scenarios))

        # Create a set from elements in remaining_scenarios close enough to this_scen
        new_atom_elements = set()
        new_atom_elements.add(this_scen)

        # Get the value of this_scen at stage t_ind + 1
        this_next_value = predicted_scenarios[this_scen][:, t_ind + 1]

        # Compute expected value of new_atom with respect to the conditional probability of the elements of the
        # atom
        mean_next_atom_value = this_next_value * probas[this_scen]

        # Compute new_atom probability which is the sum of probability of the elements of this atom
        proba_atom = probas[this_scen]

        # Going through all remaining_scenarios
        for that_scen in remaining_scenarios:
            if that_scen == this_scen:
                continue

            # Get value of that_scen at stage t_ind + 1
            that_next_value = predicted_scenarios[that_scen][:, t_ind + 1]

            # Check if that_next_value is close enough to this_next_value so that that_scenario belongs to
            # next_atom
            if np.linalg.norm(that_next_value - this_next_value) / np.linalg.norm(this_next_value) <= tol:
                new_atom_elements.add(that_scen)
                # keep computing expected value of the atom
                mean_next_atom_value += that_next_value * probas[that_scen]
                # and of its probability
                proba_atom += probas[that_scen]

        # Create new atom from new_atom_elements and set its depth equal to the depth of this_atom + 1
        new_atom = Atom(new_atom_elements, this_atom.depth() + 1, atom_id)
        atom_id += 1

        # Update value of new_atom
        new_atom.set_value(mean_next_atom_value / proba_atom)

        # Update parent of new_atom
        new_atom.set_parent(this_atom)

        # Update proba of new_atom
        new_atom.set_proba(proba_atom)

        # Add new_atom to the list of _children
        children_atoms.append(new_atom)

        # Remove scenarios in new_atom from the set of remaining scenarios
        remaining_scenarios = remaining_scenarios.difference(new_atom.elements())

    return children_atoms


def get_optimal_control_and_next_atoms(t_ind, scenario_value, predicted_scenarios, non_anticipative_controls,
                                       potential_atoms):
    if t_ind == 0:
        return non_anticipative_controls[0, 0], potential_atoms[0]._children, potential_atoms[0]
    else:
        best_dist = np.inf
        best_atom = None
        best_scen = None
        for atom in potential_atoms:
            for this_scen in atom:
                this_dist = np.abs(predicted_scenarios[this_scen][t_ind] - scenario_value)
                if this_dist < best_dist:
                    best_dist, best_atom, best_scen = this_dist, atom, this_scen
    return non_anticipative_controls[best_scen, t_ind], best_atom.children, best_atom


def retrieve_scenario_from_filtration_bfs(filtration: list[list[Atom]]):
    root = filtration[0][0]
    shape_scenario = (len(root.value()), len(filtration))
    scenario_numbers = root.size()
    list_scenarios = [np.zeros(shape_scenario) for _ in range(scenario_numbers)]
    probas = [0. for _ in range(scenario_numbers)]
    explored = set()
    queue = deque([root])
    while len(queue) > 0:
        atom = queue.popleft()
        explored.add(atom)
        for scen in atom:
            list_scenarios[scen][:, atom._depth] = atom.value()
        if atom._children is not None:
            for atom_child in atom._children:
                if atom_child not in explored:
                    queue.append(atom_child)
        else:
            scen = next(iter(atom.elements()))
            probas[scen] = atom.proba()

    return list_scenarios, probas


def retrieve_scenario_from_filtration_dfs(filtration: list[list[Atom]]):
    """
    This function retrieves scenarios and their probability from the filtration in argument by a _depth-first-search like
    algorithm
    :param filtration: As a List of List of Atom
    :return: List of Scenarios and their probability
    """
    # The filtration being a tree, its root is the first element of the first list
    root = filtration[0][0]

    # Computing the shape of the scenario, i.e. the size of the random vector and the number of stages
    shape_scenario = (len(root.value()), len(filtration))

    # Initialiazing the list of scenario and their probabilities
    list_scenarios, probas = [], []

    # Initializing the set of Atom already explored
    explored = set()

    # Add the root node to the set of explored nodes
    explored.add(root)

    # Initialize the value of the current scenario
    this_scenario = []

    # Start recursive DFS to update list_scenarios and probas
    _dfs_filtration(root, explored, list_scenarios, probas, this_scenario, shape_scenario)

    return list_scenarios, probas


def _dfs_filtration(atom, explored, list_scenarios, probas, this_scenario, shape_scenario):
    """

    :param atom: Atom being explored
    :param explored: Set of atoms already explored
    :param list_scenarios: List of scenario to update
    :param probas: Corresponding probabilities
    :param this_scenario: Current scenario being computed while traversing the tree _depth-first
    :param shape_scenario: Shape of the scenario
    """

    # If the current scenario is of maximum length before adding atom's value, this mean the DFS algorithm has
    # explored a full path from root to leave and that we are now exploring the latest unexplored fork of that path
    # This is why we initialize this-scenario as having the same atom._depth first values as the last complete scenario
    if len(this_scenario) == shape_scenario[1]:
        this_scenario = [list_scenarios[-1][:, i].reshape((shape_scenario[0], 1)) for i in range(atom._depth)]

    # Add atom's value to this-scenario
    this_scenario.append(atom.value().reshape((shape_scenario[0], 1)))

    # Recursively explore _children. If there are none, then we are at a leaf node and we can update both the list
    # of scenario by adding this scenario and the list of probability which is the probability of this atom.
    if atom._children is not None:
        for child_atom in atom._children:
            if child_atom not in explored:
                explored.add(child_atom)
                _dfs_filtration(child_atom, explored, list_scenarios, probas, this_scenario, shape_scenario)
    else:
        probas.append(atom.proba())
        list_scenarios.append(np.concatenate(this_scenario, axis=1))

import itertools, math
def aggregate_random_process(list_scenarios: list[list[np.ndarray]], list_probas=None):
    ntimes = len(list_scenarios[0][0])
    dim_random_process = len(list_scenarios)
    nscens = [len(list_scenarios[i]) for i in range(dim_random_process)]
    aggregated_scenarios = []
    if list_probas is not None:
        probas = []
    else:
        probas = None
    for scen_numbers in itertools.product(*[range(nscens[i]) for i in range(len(nscens))]):
        this_scenario = np.zeros((dim_random_process, ntimes))
        if list_probas is not None:
            this_proba = math.prod([list_probas[i][scen_numbers[i]] for i in range(dim_random_process)])
            probas.append(this_proba)
        for random_process_index in range(dim_random_process):
            this_scenario[random_process_index, :] = list_scenarios[random_process_index][
                scen_numbers[random_process_index]]
        aggregated_scenarios.append(this_scenario)
    return aggregated_scenarios, probas








