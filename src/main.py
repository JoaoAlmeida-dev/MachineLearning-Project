import copy
import os
import random
from typing import List

import mlxtend
import numpy
from numpy import ndarray
from pandas import DataFrame
from sklearn import model_selection
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
from src.constants.Constants import RED_CSV, WHITE_CSV, MIN_MAX, CORRELATIONS, TARGET
from src.dataTreatment.DataDiscretization import discretize
from src.dataTreatment.DataNormalization import normalize_set_log, normalize_set_range, normalize_standard_scaler
from src.dataTreatment.DataReduction import reduce
from src.helper.Correlation import correlate
from src.helper.RandomRemoval import random_removal_mean
from src.loader import CsvLoader
# from src.models.WineSet import WineSet
from src.plotting.WinePlot import plot_hist_wine_set, plot_wine_set, plot_decision_regions_algo, plot_means_red_white

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

    features: ndarray = numpy.asarray(wine_set.drop(labels=TARGET, axis=1))
    labels: ndarray = numpy.asarray(wine_set[TARGET].values.tolist())

    features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                test_size=0.2)
    labels_train = labels_train.ravel()

    # region Supervised
    knn, knn_accuracy = KNNClass.run(features_train, features_test, labels_train, labels_test)
    decisionTree, decisionTree_accuracy = DecisionTreeClass.run(features_train, features_test, labels_train,
                                                                labels_test)
    mlp, mlp_accuracy = MultiLayerPercetronClass.run(features_train, features_test, labels_train, labels_test)

    if PLOT:
        plot_decision_regions_algo(wine_set=wine_set, wine_set_title=title, algorithm=knn, algorithm_name="KNN")
        plot_decision_regions_algo(wine_set=wine_set, wine_set_title=title, algorithm=decisionTree,
                                   algorithm_name="DecisionTree")
        plot_decision_regions_algo(wine_set=wine_set, wine_set_title=title, algorithm=mlp,
                                   algorithm_name="MLPClassifier")

    # endregion
    # region Unsupervised
    agglomerative, agglomerative_accuracy = AgglomerativeHierarchicalClusteringClass.run(wine_set=wine_set)
    dbscan, dbscan_accuracy = DBScanClass.run(wine_set=wine_set)
    kmeans, kmeans_accuracy = KMeansClass.run(wine_set=wine_set)

    if PLOT:
        plot_decision_regions_algo(wine_set=wine_set, wine_set_title=title, algorithm=agglomerative,
                                   algorithm_name="Agglomerative")
        plot_decision_regions_algo(wine_set=wine_set, wine_set_title=title, algorithm=dbscan, algorithm_name="DBSCAN")
        plot_decision_regions_algo(wine_set=wine_set, wine_set_title=title, algorithm=kmeans, algorithm_name="Kmeans")
    # endregion

    #my_path = os.path.join(os.path.abspath(__file__), "..", "plotting", "res", title)
    #file_name = os.path.join(my_path, title + "_results.txt")
    #if not os.path.exists(my_path):
        #os.makedirs(my_path)

    file_name = os.path.join("results.csv")
    f = open(file_name, "a")
    # f.write(
    #    "knn_accuracy," + str(knn_accuracy) + "\n"+
    #    "decisionTree_accuracy," + str(decisionTree_accuracy) + "\n"+
    #    "mlp_accuracy," + str(mlp_accuracy) + "\n"+
    #    "agglomerative_accuracy," + str(agglomerative_accuracy) + "\n"+
    #    "dbscan_accuracy," + str(dbscan_accuracy) + "\n"+
    #    "kmeans_accuracy," + str(kmeans_accuracy) + "\n"
    # )
    f.write(
        title +"," + str("%.2f" % knn_accuracy) + "," + str("%.2f" % decisionTree_accuracy) + "," + str("%.2f" % mlp_accuracy) + "," + str(
            "%.2f" % agglomerative_accuracy) + "," + str("%.2f" % dbscan_accuracy) + "," + str("%.2f" % kmeans_accuracy) + ",\n"
    )
    f.close()

    print(spacer, "ENDRUN", title, spacer)


def main():
    random.seed(1)
    min_max_values = CsvLoader.load_raw_dataframe('%s' % MIN_MAX, index_col=0)

    wine_set_list: List[(DataFrame, str)] = []

    wine_set_red: DataFrame = CsvLoader.load_raw_dataframe('%s' % RED_CSV)
    wine_set_white: DataFrame = CsvLoader.load_raw_dataframe('%s' % WHITE_CSV)

    if ORIGINAL:
        original_wine_set_red = copy.deepcopy(wine_set_red)
        original_wine_set_white = copy.deepcopy(wine_set_white)

        wine_set_list.append((original_wine_set_red, "original_wine_set_red"))
        wine_set_list.append((original_wine_set_white, "original_wine_set_white"))

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
        normalize_standard_scaler(wine_set=red_normalize_mean)
        white_normalize_mean = copy.deepcopy(wine_set_white)
        normalize_standard_scaler(wine_set=white_normalize_mean)
        #plot_hist_wine_set(red_normalize_mean, plt_figure_name="red_normalize_standard_scaler", bins=30)

        wine_set_list.append((red_normalize_log, "red_normalize_log"))
        wine_set_list.append((red_normalize_range, "red_normalize_standard_scaler"))
        wine_set_list.append((red_normalize_mean, "red_normalize_mean"))
        wine_set_list.append((white_normalize_log, "white_normalize_log"))
        wine_set_list.append((white_normalize_range, "white_normalize_standard_scaler"))
        wine_set_list.append((white_normalize_mean, "white_normalize_mean"))

    if DISCRETIZE:
        discretized_set_red = copy.deepcopy(wine_set_red)
        discretize(wine_set=discretized_set_red, num_bins=3)
        wine_set_list.append((discretized_set_red, "discretized_set_red"))

        discretized_set_white = copy.deepcopy(wine_set_white)
        discretize(wine_set=discretized_set_white, num_bins=3)
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

    if PLOT:
        plot_means_red_white(wine_set_red=wine_set_red, wine_set_white=wine_set_white, )

        plot_wine_set(wine_set_red, plt_figure_name="wine_set_red_plot")
        plot_wine_set(wine_set_white, plt_figure_name="wine_set_white_plot")
        plot_hist_wine_set(wine_set_red, "wine_set_red", bins=30)
        plot_hist_wine_set(wine_set_white, "wine_set_white", bins=30)

    if ALGO:
        # make a dataframe with algorithm as rows and columns as type of treatment
        file_name = os.path.join("results.csv")
        f = open(file_name, "w")
        f.write("index,knn,decisionTree,mlp,agglomerative,dbscan,kmeans,\n")
        f.close()

        for set in wine_set_list:
            run_algos(wine_set=set[0], title=set[1])


if __name__ == '__main__':
    main()
