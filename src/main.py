import copy
import random
from typing import List

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
from src.helper.RandomRemoval import random_removal_mean
from src.loader import CsvLoader
# from src.models.WineSet import WineSet
from src.plotting.WinePlot import plot_hist_wine_set, plot_wine_set

ALGO = True
PLOT = False
NORMALIZE = True
DISCRETIZE = True
REDUCE = True
REMOVE = True
ORIGINAL = True


def run_algos(wine_set: DataFrame, title: str):
    spacer: str = "=========================="
    print(spacer, "RUNNING", title, spacer)
    # region Supervised
    knn: KNeighborsClassifier = KNNClass.run(wine_set=wine_set)
    decisionTree: DecisionTreeClassifier = DecisionTreeClass.run(wine_set=wine_set)
    mlp: MLPClassifier = MultiLayerPercetronClass.run(wine_set=wine_set)
    # endregion
    # region Unsupervised
    agglomerative = AgglomerativeHierarchicalClusteringClass.run(wine_set=wine_set)
    dbscan: DBSCAN = DBScanClass.run(wine_set=wine_set)
    kmeans: KMeans = KMeansClass.run(wine_set=wine_set)
    print(spacer, "ENDRUN", title, spacer)


# endregion

def plot_raw_dataset():
    wine_set_red: DataFrame = CsvLoader.load_dataframe('%s' % RED_CSV)
    wine_set_white: DataFrame = CsvLoader.load_dataframe('%s' % WHITE_CSV)
    plot_hist_wine_set(wine_set_red, "wine_set_red", bins=30)
    plot_hist_wine_set(wine_set_white, "wine_set_white", bins=30)


def main():
    random.seed(1)
    min_max_values = CsvLoader.load_raw_dataframe('%s' % MIN_MAX, index_col=0)

    wine_set_list: List[(DataFrame, str)] = []

    wine_set_red: DataFrame = CsvLoader.load_dataframe('%s' % RED_CSV)
    wine_set_white: DataFrame = CsvLoader.load_dataframe('%s' % WHITE_CSV)

    if ORIGINAL:
        original_wine_set_red = copy.deepcopy(wine_set_red)
        original_wine_set_white = copy.deepcopy(wine_set_white)

        wine_set_list.append((original_wine_set_red, "original_wine_set_red"))
        wine_set_list.append((original_wine_set_white, "original_wine_set_white"))

    if PLOT:
        plot_wine_set(wine_set_red, plt_figure_name="wine_set_red_plot")
        plot_raw_dataset()

    if REMOVE:
        removed_10_set_red = copy.deepcopy(wine_set_red)
        random_removal_mean(dataset=removed_10_set_red, removal_percentage=0.1)
        removed_20_set_red = copy.deepcopy(wine_set_red)
        random_removal_mean(dataset=removed_20_set_red, removal_percentage=0.2)
        removed_30_set_red = copy.deepcopy(wine_set_red)
        random_removal_mean(dataset=removed_30_set_red, removal_percentage=0.3)

        removed_10_set_white = copy.deepcopy(wine_set_white)
        random_removal_mean(dataset=removed_10_set_white, removal_percentage=0.1)
        removed_20_set_white = copy.deepcopy(wine_set_white)
        random_removal_mean(dataset=removed_20_set_white, removal_percentage=0.2)
        removed_30_set_white = copy.deepcopy(wine_set_white)
        random_removal_mean(dataset=removed_30_set_white, removal_percentage=0.3)

        wine_set_list.append((removed_10_set_red, "removed_10_set_red"))
        wine_set_list.append((removed_20_set_red, "removed_20_set_red"))
        wine_set_list.append((removed_30_set_red, "removed_30_set_red"))
        wine_set_list.append((removed_10_set_white, "removed_10_set_white"))
        wine_set_list.append((removed_20_set_white, "removed_20_set_white"))
        wine_set_list.append((removed_30_set_white, "removed_30_set_white"))

    if NORMALIZE:
        # normalize_set_plot(wine_set=wine_set_red, min_max_values=min_max_values, title="wine_set_red")

        red_normalize_log = copy.deepcopy(wine_set_red)
        normalize_set_log(wine_set=red_normalize_log)
        white_normalize_log = copy.deepcopy(wine_set_white)
        normalize_set_log(wine_set=white_normalize_log)

        red_normalize_range = copy.deepcopy(wine_set_red)
        normalize_set_range(wine_set=red_normalize_range, range_dataframe=min_max_values, set_name="red")
        white_normalize_range = copy.deepcopy(wine_set_white)
        normalize_set_range(wine_set=white_normalize_range, range_dataframe=min_max_values, set_name="white")

        red_normalize_mean = copy.deepcopy(wine_set_red)
        normalize_set_mean(wine_set=red_normalize_mean)
        white_normalize_mean = copy.deepcopy(wine_set_white)
        normalize_set_mean(wine_set=white_normalize_mean)

        wine_set_list.append((red_normalize_log, "red_normalize_log"))
        wine_set_list.append((red_normalize_range, "red_normalize_range"))
        wine_set_list.append((red_normalize_mean, "red_normalize_mean"))
        wine_set_list.append((white_normalize_log, "white_normalize_log"))
        wine_set_list.append((white_normalize_range, "white_normalize_range"))
        wine_set_list.append((white_normalize_mean, "white_normalize_mean"))

    if DISCRETIZE:
        discretized_set_red = copy.deepcopy(wine_set_red)
        discretize(wine_set=discretized_set_red, num_bins=5)
        wine_set_list.append((discretized_set_red, "discretized_set_red"))

        discretized_set_white = copy.deepcopy(wine_set_white)
        discretize(wine_set=discretized_set_white, num_bins=5)
        wine_set_list.append((discretized_set_white, "discretized_set_white"))

    if REDUCE:
        reduced_set_red = copy.deepcopy(wine_set_red)
        reduced_set_correlations_red = correlate(reduced_set_red)
        reduce(wine_set=reduced_set_red, correlations=reduced_set_correlations_red)
        wine_set_list.append((reduced_set_red, "reduced_set_red"))

        reduced_set_white = copy.deepcopy(wine_set_white)
        reduced_set_correlations_white = correlate(reduced_set_red)
        reduce(wine_set=reduced_set_white, correlations=reduced_set_correlations_white)
        wine_set_list.append((reduced_set_white, "reduced_set_white"))

    if ALGO:
        # make a dataframe with algorithm as rows and columns as tipe of treatment
        for set in wine_set_list:
            run_algos(wine_set=set[0], title=set[1])


if __name__ == '__main__':
    main()
