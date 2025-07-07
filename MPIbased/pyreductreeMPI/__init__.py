"""
REDUCTION_NUMBA Package

Ce package fournit des outils pour la construction, la réduction et l'analyse d'arbres dans le contexte
de l'optimisation stochastique.

Modules principaux :
- create_homogenous_filtrations_and_trees : Génération et manipulation des arbres (homogènes).
- boostedKP : Package de réduction d'arbres et calculs de distances nested.
    - KP_reduction est l'algorithme de réduction d'arbre, Nested_Distance calcule la ND entre 2 arbres
"""
from .boostedKP import KP_reduction, Nested_Distance
from .filtrations_and_trees import generate_uniform_tree, filtration_into_networkx_tree
