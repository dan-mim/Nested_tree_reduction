from pyreductreeMPI import generate_uniform_tree, KP_reduction, filtration_into_networkx_tree, Nested_Distance

# Generate filtrations
original_filtration = generate_uniform_tree(typearbre=3, seed=0, depth=6)
reduced_filtration = generate_uniform_tree(typearbre=2, seed=0, depth=6)

# Turn filtration into trees (networkx)
H = filtration_into_networkx_tree(original_filtration)
G = filtration_into_networkx_tree(reduced_filtration)
# Visualize
from pyreductreeMPI.boostedKP.visualization_tree import *
draw_tree(H)

# REDUCTION
# Filtration based Reduction
res_0 = KP_reduction(original_filtration, reduced_filtration,
                     method='LP',
                     delta=1000,  itred=7, npool=1,
                     lambda_IBP=100, rho=1000, precisionMAM=10**-4)
reduced_filtration2 = res_0['reduced_filtration']

# COMPARE distances
# Visualize
reduced_tree = filtration_into_networkx_tree(reduced_filtration2)
draw_tree(reduced_tree)
# Compute nested distance between the original and the reduced tree
dist_filtr2 = Nested_Distance(original_filtration, reduced_filtration2)
print(dist_filtr2)