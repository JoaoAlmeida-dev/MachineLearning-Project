import random

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.algorithms.supervised.DecisionTreeClass import DecisionTreeClass
from src.algorithms.supervised.MultiLayerPercetronClass import MultiLayerPercetronClass
from src.algorithms.unsupervised.AgglomerativeHierarchicalClusteringClass import \
    AgglomerativeHierarchicalClusteringClass
from src.algorithms.unsupervised.DBScanClass import DBScanClass
from src.algorithms.unsupervised.KMeansClass import KMeansClass
from src.constants.Constants import RED_CSV
from src.algorithms.supervised.KNNClass import KNNClass
from src.dataTreatment.RandomRemoval import random_removal_mean
from src.loader import CsvLoader
from src.models.WineSet import WineSet
from src.plotting import WinePlot
from src.plotting.WinePlot import plot_wine_set


def main():
    random.seed(1)


    algo = False
    plot = True
    # wine_set: WineSet = CsvLoader.load_List('%s' % RED_CSV, skip_header=True)
    wine_set: WineSet = CsvLoader.load_dataframe('%s' % RED_CSV, skip_header=True)
    wine_set2: WineSet = CsvLoader.load_dataframe('%s' % RED_CSV, skip_header=True)
    print(wine_set.wine_dataframe.head)
    print("len=", len(wine_set))
    random_removal_mean(dataset=wine_set, removal_percentage=0.1)
    plot_wine_set(wine_set)

    random_removal_mean(dataset=wine_set2, removal_percentage=0.9)
    plot_wine_set(wine_set2)



    if algo:
        # region Supervised

        knn: KNeighborsClassifier = KNNClass.run(wine_set=wine_set)
        decisionTree: DecisionTreeClassifier = DecisionTreeClass.run(wine_set=wine_set)
        mlp: MLPClassifier = MultiLayerPercetronClass.run(wine_set=wine_set)

        # endregion

        # region Unsupervised

        agglomerative = AgglomerativeHierarchicalClusteringClass.run(wine_set=wine_set)
        dbscan: DBSCAN = DBScanClass.run(wine_set=wine_set)
        kmeans: KMeans = KMeansClass.run(wine_set=wine_set)
        # endregion


# plotWineSet(wine_set)


if __name__ == '__main__':
    main()
