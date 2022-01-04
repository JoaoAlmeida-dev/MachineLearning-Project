import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.algorithms.supervised.DecisionTree import DecisionTree
from src.algorithms.supervised.MultiLayerPercetron import MultiLayerPercetron
from src.algorithms.unsupervised.AgglomerativeHierarchicalClustering import AgglomerativeHierarchicalClustering
from src.algorithms.unsupervised.DBScan import DBScan
from src.constants.Constants import RED_CSV
from src.algorithms.supervised.KNN import KNN
from src.dataTreatment.RandomRemoval import RandomRemoval
from src.loader import CsvLoader
from src.models.WineSet import WineSet


def main():
    wine_set: WineSet = CsvLoader.load('%s' % RED_CSV, skip_header=True)

    # print("len=", len(wine_set))
    RandomRemoval(dataset=wine_set, removal_percentage=0.1)
    # print("len=", len(wine_set))

    # region Supervised

    knn: KNeighborsClassifier = KNN.run(wine_set=wine_set)
    decisionTree: DecisionTreeClassifier = DecisionTree.run(wine_set=wine_set)
    mlp: MLPClassifier = MultiLayerPercetron.run(wine_set=wine_set)

    # endregion

    # region Unsupervised

    agglomerative = AgglomerativeHierarchicalClustering.run(wine_set=wine_set)
    dbscan: DBSCAN = DBScan.run(wine_set=wine_set)

    # endregion


# plotWineSet(wine_set)


if __name__ == '__main__':
    main()
