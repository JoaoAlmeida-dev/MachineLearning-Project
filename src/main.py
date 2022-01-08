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
from src.constants.Constants import RED_CSV, WHITE_CSV, MIN_MAX, CATEGORIES
from src.dataTreatment.DataDiscretization import discretize
from src.dataTreatment.DataNormalization import normalize_set_log, normalize_set_range, normalize_set_mean
from src.helper.Correlation import correlate
from src.loader import CsvLoader
from src.models.WineSet import WineSet
from src.plotting.WinePlot import plot_hist_wine_set, plot_wine_set


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


def normalize_set_plot(wine_set: WineSet, min_max_values: DataFrame, title: str):
    wine_set_log = copy.deepcopy(wine_set)
    wine_set_range = copy.deepcopy(wine_set)
    wine_set_mean = copy.deepcopy(wine_set)

    normalize_set_log(wine_set=wine_set_log)
    print("wine_set_log\n", wine_set_log.wine_dataframe.head())
    plot_hist_wine_set(wine_set=wine_set_log, plt_figure_name=title + "-log", bins=20)

    normalize_set_range(wine_set=wine_set_range, range_dataframe=min_max_values, set_name="red")
    print("wine_set_range\n", wine_set_range.wine_dataframe.head())
    plot_hist_wine_set(wine_set=wine_set_range, plt_figure_name=title + "-range", bins=30)

    normalize_set_mean(wine_set=wine_set_mean)
    print("wine_set_mean\n", wine_set_mean.wine_dataframe.head())
    plot_hist_wine_set(wine_set=wine_set_mean, plt_figure_name=title + "-mean", bins=30)


def main():
    algo = False
    plot_bool = True
    normalize_and_plot_bool = True
    discretize_and_plot_bool = True
    correlate_bool = True

    random.seed(1)
    min_max_values = CsvLoader.load_raw_dataframe('%s' % MIN_MAX, index_col=0)
    categories = CsvLoader.load_raw_dataframe('%s' % CATEGORIES)

    print(min_max_values.head())
    print(categories.head())
    wine_set_red: WineSet = CsvLoader.load_dataframe('%s' % RED_CSV)
    wine_set_white: WineSet = CsvLoader.load_dataframe('%s' % WHITE_CSV)

    # random_removal_mean(dataset=wine_set_red, removal_percentage=0.1)
    if correlate_bool:
        correlate(wine_set_red)

    if plot_bool:
        plot_wine_set(wine_set_red, plt_figure_name="wine_set_red_plot")
        plot_raw_dataset()

    if normalize_and_plot_bool:
        normalize_set_plot(wine_set=wine_set_red, min_max_values=min_max_values, title="wine_set_red")

    if discretize_and_plot_bool:
        wine_set_red = discretize(wine_set=wine_set_red, categories=categories, num_bins=5)

    if algo:
        run_algos(wine_set=wine_set_red)


if __name__ == '__main__':
    main()
