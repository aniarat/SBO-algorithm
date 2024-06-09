from enum import Enum


class CrossingMethodsBin(Enum):
    NONE = -1
    TEST = -2
    SINGLE_POINT = 0
    DOUBLE_POINT = 1
    TRIPLE_POINT = 2
    UNIFORM = 3
    GRAIN = 4
    SCANNING = 5
    PARTIAL = 6
    MULTIVARIATE = 7
    BUILD_IN_SINGLE_POINT = 'single_point'
    BUILD_IN_DOUBLE_POINT = 'two_points'
    BUILD_IN_UNIFORM = 'uniform'
    SINGLE_POINT_STRING = 'Krzyżowanie 1 punktowe'
    DOUBLE_POINT_STRING = 'Krzyżowanie 2 punktowe'
    TRIPLE_POINT_STRING = 'Krzyżowanie 3 punktowe'
    UNIFORM_STRING = 'Krzyżowanie jednorodne'
    GRAIN_STRING = 'Krzyżowanie ziarniste'
    SCANNING_STRING = 'Krzyżowanie skanujące'
    PARTIAL_STRING = 'Krzyżowanie częściowe'
    MULTIVARIATE_STRING = 'Krzyżowanie wielowymiarowe'

    ALL_OPTIONS_STRING = [SINGLE_POINT_STRING,
                          DOUBLE_POINT_STRING,
                          TRIPLE_POINT_STRING,
                          UNIFORM_STRING,
                          GRAIN_STRING,
                          SCANNING_STRING,
                          PARTIAL_STRING,
                          MULTIVARIATE_STRING]


class CrossingMethodsDec(Enum):
    NONE = -1
    TEST = -2
    SINGLE_POINT_ARITHMETIC = 0
    SINGLE_POINT_ARITHMETIC_STRING = 'Krzyżowanie 1 arytmetyczne'
    ARITHMETIC = 1
    ARITHMETIC_STRING = 'Krzyżowanie arytmetyczne'
    LINEAR = 2
    LINEAR_STRING = 'Krzyżowanie linearne'
    BLEND_ALFA = 3
    BLEND_ALFA_STRING = 'Krzyżowanie mieszające typu alfa'
    BLEND_ALFA_BETA = 4
    BLEND_ALFA_BETA_STRING = 'Krzyżowanie mieszające typu alfa i beta'
    AVERAGE = 5
    AVERAGE_STRING = 'Krzyżowanie uśredniające'
    SIMPLE = 6
    SIMPLE_STRING = 'Krzyżowanie proste'
    RANDOM = 7
    RANDOM_STRING = 'Krzyżowanie przypadkowe'

    ALL_OPTIONS_STRING = [SINGLE_POINT_ARITHMETIC_STRING, ARITHMETIC_STRING, LINEAR_STRING, BLEND_ALFA_STRING,
                          BLEND_ALFA_BETA_STRING, AVERAGE_STRING, SIMPLE_STRING, RANDOM_STRING]


class MinMax(Enum):
    MIN = 0
    MAX = 1


class FunctionsOptions(Enum):
    RASTRIGIN = 0
    SCHWEFEK = 1
