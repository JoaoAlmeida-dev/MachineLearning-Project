import copy
import random

from pandas import DataFrame
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.algorithms.supervised.DecisionTreeClass import DecisionTreeClass
from src.algorithms.supervised.KNNClass import KNNClass
from src.algorithms.supervised.MultiLayerPercetronClass import MultiLayerPercetronClass
from src.algorithms.unsupervised.AgglomerativeHierarchicalClusteringClass import \
    AgglomerativeHierarchicalClusteringClass
from src.algorithms.unsupervised.DBScanClass import DBScanClass
from src.algorithms.unsupervised.KMeansClass import KMeansClass
from src.constants.Constants import RED_CSV, WHITE_CSV, MIN_MAX, CORRELATIONS
from src.dataTreatment.DataDiscretization import discretize
from src.dataTreatment.DataNormalization import normalize_set_log, normalize_set_range, normalize_set_mean
from src.dataTreatment.DataReduction import reduce
from src.helper.Correlation import correlate
from src.loader import CsvLoader
from src.models.WineSet import WineSet
from src.plotting.WinePlot import plot_hist_wine_set, plot_wine_set

ALGO = False
PLOT = False
NORMALIZE = False
DISCRETIZE = False
REDUCE = True
REMOVE = True
ORIGINAL = True

def run_algos(wine_set: WineSet):
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

def plot_raw_dataset():
    wine_set_red: WineSet = CsvLoader.load_dataframe('%s' % RED_CSV)
    wine_set_white: WineSet = CsvLoader.load_dataframe('%s' % WHITE_CSV)
    plot_hist_wine_set(wine_set_red, "wine_set_red", bins=30)
    plot_hist_wine_set(wine_set_white, "wine_set_white", bins=30)


def main():

    random.seed(1)
    min_max_values = CsvLoader.load_raw_dataframe('%s' % MIN_MAX, index_col=0)

    wine_set_list: List[WineSet] = []

    wine_set_red: WineSet = CsvLoader.load_dataframe('%s' % RED_CSV)
    wine_set_white: WineSet = CsvLoader.load_dataframe('%s' % WHITE_CSV)

    if ORIGINAL:
        original_wine_set_red = copy.deepcopy(wine_set_red)
        wine_set_list.append(wine_set_red)

        original_wine_set_white = copy.deepcopy(wine_set_white)
        wine_set_list.append(wine_set_white)

    if PLOT:
        plot_wine_set(wine_set_red, plt_figure_name="wine_set_red_plot")
        plot_raw_dataset()

    if NORMALIZE_AND_PLOT_BOOL:
        normalize_set_plot(wine_set=wine_set_red, min_max_values=min_max_values, title="wine_set_red")

    if DISCRETIZE:
        discretized_set_red = copy.deepcopy(wine_set_red)
        discretize(wine_set=discretized_set_red, num_bins=5)

        discretized_set_white = copy.deepcopy(wine_set_white)
        discretize(wine_set=discretized_set_white, num_bins=5)

    if REDUCE:
        reduced_set_red = copy.deepcopy(wine_set_red)
        reduced_set_correlations_red = correlate(reduced_set_red)
        reduce(wine_set=reduced_set_red, correlations=reduced_set_correlations_red)


        reduced_set_white = copy.deepcopy(wine_set_white)
        reduced_set_correlations_white = correlate(reduced_set_red)
        reduce(wine_set=reduced_set_white, correlations=reduced_set_correlations_white)

    if ALGO:
        run_algos(wine_set=wine_set_red)


if __name__ == '__main__':
    main()
