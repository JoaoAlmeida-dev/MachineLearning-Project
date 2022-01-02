import csv
from typing import List

from src.models.Wine import Wine


def load(filename: str, skip_header: bool = True) -> List[Wine]:
    wine_list: List[Wine] = []

    # opening the CSV file
    with open(filename, mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)
        if skip_header: next(csvFile)
        # displaying the contents of the CSV file
        for lines in csvFile:
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, output = lines[0].split(";")
            wine = Wine(fixed_acidity=fixed_acidity, volatile_acidity=volatile_acidity,
                        citric_acid=citric_acid, residual_sugar=residual_sugar, chlorides=chlorides,
                        free_sulfur_dioxide=free_sulfur_dioxide,
                        total_sulfur_dioxide=total_sulfur_dioxide, density=density, pH=pH, sulphates=sulphates,
                        alcohol=alcohol, output=output)
            wine_list.append(wine)

    return wine_list
