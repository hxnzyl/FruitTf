from os.path import join, dirname, abspath

ROOT_PATH: str = dirname(abspath(__file__))
DATA_PATH: str = join(ROOT_PATH, 'data/')

TRAIN_FRUIT_PATH: str = join(DATA_PATH, 'fruits/train/')
TEST_FRUIT_PATH: str = join(DATA_PATH, 'fruits/test/')

WEIGHTS_PATH: str = join(DATA_PATH, 'weights/')
MODELS_PATH: str = join(DATA_PATH, 'models/')
RESULTS_PATH: str = join(DATA_PATH, 'results/')

# 96, 128, 160, 192, 224
FRUIT_IMAGE_WIDTH: int = 128
FRUIT_IMAGE_HEIGHT: int = 128

FRUIT_LABEL_LIST: list[str] = ['apple', 'banana', 'bayberry', 'blueberry', 'cantaloupe', 'cherry', 'grape', 'lemon',
                               'longan', 'mango', 'orange', 'peach', 'pear', 'strawberry']
