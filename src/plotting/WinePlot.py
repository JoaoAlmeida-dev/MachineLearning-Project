import matplotlib.pyplot as plt
import mlxtend
import numpy as np
from mlxtend.plotting import plot_decision_regions
from pandas import DataFrame
import scikitplot as skplt
import os

from src.constants.Constants import HEADERS, FEATURES, N_VARIABLES, TARGET


def plot_hist_wine_set(wine_set: DataFrame, plt_figure_name: str, bins: int):

    my_path = os.path.join(os.path.abspath(__file__), "..", "plot", wine_set_title, algorithm_name,)
    file_name = plt_figure_name
    if not os.path.exists(my_path):
        os.makedirs(my_path)

    row_count = int(N_VARIABLES ** 0.5) + 1
    fig = plt.gcf()
    fig.canvas.set_window_title(plt_figure_name)

    for header_index, header_string in enumerate(HEADERS):
        plt.subplot(row_count, row_count, header_index + 1)
        plt.title(header_string)
        wine_set.loc[:, header_string].plot.hist(bins=bins)

    plt.savefig(file_name)

    plt.tight_layout()
    plt.show()


def plot_wine_set(wine_set: DataFrame, plt_figure_name: str):
    wine_set.plot()
    plt.title(plt_figure_name)
    plt.tight_layout()
    plt.show()


def plot_by_features(wine_set: DataFrame, plt_figure_name: str):
    row_count = int(len(FEATURES) ** 0.5) + 1
    fig = plt.gcf()
    fig.canvas.set_window_title(plt_figure_name)

    for header_index, header_string in enumerate(HEADERS):
        plt.subplot(row_count, row_count, header_index + 1)
        plt.title(header_string)
        wine_set.loc[:, header_string].plot.hist(bins=2)

    plt.tight_layout()
    plt.show()


def plot_decision_regions_algo(wine_set: DataFrame, algorithm, algorithm_name: str, wine_set_title: str):
    my_path = os.path.join(os.path.abspath(__file__), "..", "res", wine_set_title, algorithm_name,)
    file_name = wine_set_title+"_"+algorithm_name
    if not os.path.exists(my_path):
        os.makedirs(my_path)


    features_df: DataFrame = wine_set.drop(labels=TARGET, axis=1)
    labels_df: DataFrame = wine_set[TARGET]

    try:
        plt.title(algorithm_name + " decision_regions")

        plot_decision_regions(features_df.values, labels_df.values, clf=algorithm, legend=2, feature_index=[0, 1],
                              filler_feature_values={2: wine_set.iloc[:, 2].mean(), 3: wine_set.iloc[:, 3].mean(),
                                                     4: wine_set.iloc[:, 4].mean(), 5: wine_set.iloc[:, 5].mean(),
                                                     6: wine_set.iloc[:, 6].mean(), 7: wine_set.iloc[:, 7].mean(),
                                                     8: wine_set.iloc[:, 8].mean(), 9: wine_set.iloc[:, 9].mean(),
                                                     10: wine_set.iloc[:, 10].mean()
                                                     },
                              filler_feature_ranges={2: wine_set.iloc[:, 2].max(), 3: wine_set.iloc[:, 3].max(),
                                                     4: wine_set.iloc[:, 4].max(), 5: wine_set.iloc[:, 5].max(),
                                                     6: wine_set.iloc[:, 6].max(), 7: wine_set.iloc[:, 7].max(),
                                                     8: wine_set.iloc[:, 8].max(), 9: wine_set.iloc[:, 9].max(),
                                                     10: wine_set.iloc[:, 10].max()})

        #plt.savefig(cwd_path +"\\res\\" + wine_set_title + "\\" + algorithm_name + "\\DecisionRegion.svg")
        plt.savefig(os.path.join(my_path,file_name+"_DecisionRegion.svg"))
    except:
        print(algorithm_name + " cant run plot_decision_regions")
    try:
        plt.title(algorithm_name + " learning_curve")
        skplt.estimators.plot_learning_curve(clf=algorithm, X=features_df, y=labels_df, cv=7, shuffle=True)
        #plt.savefig(cwd_path + "\\res\\" + wine_set_title + "\\" + algorithm_name + "\\LearningCurve.svg")
        plt.savefig(os.path.join(my_path,file_name+"_LearningCurve.svg"))
    except:
        print(algorithm_name + " cant run plot_learning_curve")
    try:
        skplt.estimators.plot_feature_importances(algorithm, feature_names=wine_set.columns[:-1:],x_tick_rotation=90 );
        #plt.savefig(cwd_path + "\\res\\" + wine_set_title + "\\" + algorithm_name + "\\FeatureImportances.svg")
        plt.savefig(os.path.join(my_path,file_name+"_FeatureImportances.svg"))
    except:
        print(algorithm_name + " cant run plot_feature_importances")

    """filler_feature_values={2: wine_set.iloc[:, 2], 3: wine_set.iloc[:, 3],
                         4: wine_set.iloc[:, 4], 5: wine_set.iloc[:, 5],
                         6: wine_set.iloc[:, 6], 7: wine_set.iloc[:, 7],
                         8: wine_set.iloc[:, 8], 9: wine_set.iloc[:, 9],
                         10: wine_set.iloc[:, 10]
                         },
    filler_feature_ranges={2: wine_set.iloc[:, 2], 3: wine_set.iloc[:, 3],
                         4: wine_set.iloc[:, 4], 5: wine_set.iloc[:, 5],
                         6: wine_set.iloc[:, 6], 7: wine_set.iloc[:, 7],
                         8: wine_set.iloc[:, 8], 9: wine_set.iloc[:, 9],
                         10: wine_set.iloc[:, 10]})"""
