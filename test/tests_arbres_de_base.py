from pyreductree import generate_uniform_tree, KP_reduction, filtration_into_networkx_tree, nested_distance


# Generate filtrations
original_filtration = generate_uniform_tree(typearbre=3, seed=0, depth=6)
reduced_filtration = generate_uniform_tree(typearbre=2, seed=0, depth=6)

# Turn filtration into trees (networkx)
H = filtration_into_networkx_tree(original_filtration)
# Visualize
from pyreductree.visualization_tree import *
draw_tree(H)

# REDUCTION
# Filtration based Reduction
reduced_filtration2, distance_nd = KP_reduction(
    original_filtration, reduced_filtration, method='LP',
    delta=1000,  itred=7, npool=1,
    lambda_IBP=100, rho=1000,
    precisionMAM=10**-4)

# COMPARE distances
reduced_tree = filtration_into_networkx_tree(reduced_filtration2)
dist_filtr2 = nested_distance(original_filtration, reduced_filtration2)
print(dist_filtr2)

