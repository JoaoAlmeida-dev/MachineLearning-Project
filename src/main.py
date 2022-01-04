from src.algorithms.supervised.DecisinTree import DecisionTree
from src.algorithms.supervised.MultiLayerPercetron import MultiLayerPercetron
from src.constants.Constants import RED_CSV
from src.algorithms.supervised.KNN import KNN
from src.dataTreatment.RandomRemoval import RandomRemoval
from src.loader import CsvLoader
from src.models.WineSet import WineSet


def main():
    wine_set: WineSet = CsvLoader.load('%s' % RED_CSV, skip_header=True)

    #print("len=", len(wine_set))
    RandomRemoval(dataset=wine_set, removal_percentage=0.1)
    #print("len=", len(wine_set))

#region Supervised
    KNN.run(wine_set=wine_set)
    DecisionTree.run(wine_set=wine_set)
    MultiLayerPercetron.run(wine_set=wine_set)
#endregion
#region Unsupervised



#endregion
    # plotWineSet(wine_set)


if __name__ == '__main__':
    main()
