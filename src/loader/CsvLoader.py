import csv
from typing import List

from src.models.Wine import Wine
from src.models.WineSet import WineSet


def load_panda(filename: str):
    pass

def load(filename: str, skip_header: bool = True) -> WineSet:
    wine_list: List[Wine] = []

    # opening the CSV file
    with open(filename, mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)
        if skip_header: next(csvFile)
        # displaying the contents of the CSV file
        for lines in csvFile:
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, output = lines[0].split(";")
            wine = Wine(fixed_acidity=float(fixed_acidity), volatile_acidity=float(volatile_acidity),
                        citric_acid=float(citric_acid), residual_sugar=float(residual_sugar), chlorides=float(chlorides),
                        free_sulfur_dioxide=float(free_sulfur_dioxide),
                        total_sulfur_dioxide=float(total_sulfur_dioxide), density=float(density), pH=float(pH), sulphates=float(sulphates),
                        alcohol=float(alcohol), output=float(output))
            wine_list.append(wine)

    return WineSet(wine_list=wine_list)
