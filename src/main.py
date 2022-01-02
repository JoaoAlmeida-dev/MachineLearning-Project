from typing import List

from src.Loader import CsvLoader
from src.models.Wine import Wine

RED_CSV = 'Resources/winequality-red.csv'
WHITE_CSV = 'Resources/winequality-white.csv'


def main():
    wine_list: List[Wine] = CsvLoader.load('%s' % RED_CSV,skip_header=True)
    print(wine_list)


if __name__ == '__main__':
    main()
