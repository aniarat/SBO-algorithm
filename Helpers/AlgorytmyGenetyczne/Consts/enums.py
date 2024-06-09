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
    SINGLE_POINT_STRING = '1 punktowe'
    DOUBLE_POINT_STRING = '2 punktowe'
    TRIPLE_POINT_STRING = '3 punktowe'
    UNIFORM_STRING = 'jednorodne'
    GRAIN_STRING = 'ziarniste'
    SCANNING_STRING = 'skanujące'
    PARTIAL_STRING = 'częściowe'
    MULTIVARIATE_STRING = 'wielowymiarowe'

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
    SINGLE_POINT_ARITHMETIC_STRING = 'pojedyncze arytmetyczne'
    ARITHMETIC = 1
    ARITHMETIC_STRING = 'arytmetyczne'
    LINEAR = 2
    LINEAR_STRING = 'linearne'
    BLEND_ALFA = 3
    BLEND_ALFA_STRING = 'mieszające typu alfa'
    BLEND_ALFA_BETA = 4
    BLEND_ALFA_BETA_STRING = 'mieszające typu alfa i beta'
    AVERAGE = 5
    AVERAGE_STRING = 'uśredniające'
    SIMPLE = 6
    SIMPLE_STRING = 'proste'
    RANDOM = 7
    RANDOM_STRING = 'przypadkowe'

    ALL_OPTIONS_STRING = [SINGLE_POINT_ARITHMETIC_STRING, ARITHMETIC_STRING, LINEAR_STRING, BLEND_ALFA_STRING,
                          BLEND_ALFA_BETA_STRING, AVERAGE_STRING, SIMPLE_STRING, RANDOM_STRING]


class MinMax(Enum):
    MIN = 0
    MAX = 1


class FunctionsOptions(Enum):
    RASTRIGIN = 0
    SCHWEFEL = 1
