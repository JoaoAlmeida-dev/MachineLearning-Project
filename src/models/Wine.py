class Wine:
    # 1 - fixed acidity
    # 2 - volatile acidity
    # 3 - citric acid
    # 4 - residual sugar
    # 5 - chlorides
    # 6 - free_sulfur_dioxide
    # 7 - total_sulfur_dioxide
    # 8 - density
    # 9 - pH
    # 10 - sulphates
    # 11 - alcohol
    # Output variable(based on sensory data):
    # 12 - quality(score between 0 and 10)
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    output: float

    def __init__(self, fixed_acidity: float, volatile_acidity: float, citric_acid: float, residual_sugar: float,
                 chlorides: float, free_sulfur_dioxide: float, total_sulfur_dioxide: float, density: float, pH: float,
                 sulphates: float, alcohol: float, output: float = 0):
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol
        self.output = output

    def __str__(self):
        return "{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}".format(str(self.fixed_acidity), str(self.volatile_acidity),
                                                               str(self.citric_acid), str(self.residual_sugar),
                                                               str(self.chlorides), str(self.free_sulfur_dioxide),
                                                               str(self.total_sulfur_dioxide), str(self.density),
                                                               str(self.pH), str(self.sulphates), str(self.alcohol),
                                                               str(self.output))

    def __repr__(self):
        return self.__str__()
