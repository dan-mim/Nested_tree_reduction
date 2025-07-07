import pickle
from pyreductree import KP_reduction, reduce_scenarios_forward_selection
from pystocoptim import compute_filtration


def main():

    with open("test_reduc_tree.pickle", "rb") as fh:
        datas = pickle.load(fh)

    consos, prods = datas["consos"], datas["prods"]

    for ic, conso in enumerate(consos):
        consos[ic] = conso.reshape((1, len(conso)))

    for ic, prod in enumerate(prods):
        prods[ic] = prod.reshape((1, len(prod)))

    conso_filtration = compute_filtration(consos)
    prod_filtration = compute_filtration(prods)

    reduced_datas, reduced_probas = reduce_scenarios_forward_selection([consos, prods], [10, 10])

    reduced_conso_filtration = compute_filtration(reduced_datas[0], probas=reduced_probas[0])
    reduced_prod_filtration = compute_filtration(reduced_datas[1], probas=reduced_probas[1])

    filtration_conso, distance = KP_reduction(
        conso_filtration, reduced_conso_filtration, method='LP', delta=1000,  itred=7, npool=1, lambda_IBP=100,
        rho=1000, precisionMAM=10**-4
    )



if __name__ == "__main__":
    main()