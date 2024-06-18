from pathlib import Path


SHARP_PATH = Path().joinpath("images/train/sharp").as_posix()
BLUR_PATH = Path().joinpath("images/train/blur").as_posix()

IMG_SHAPE = (218, 178, 3)

BATCH_SIZE = 64
EPOCHS = 10

BASE_OUTPUT = Path().joinpath("output").as_posix()

MODEL_PATH = Path(BASE_OUTPUT).joinpath("siamese_model").as_posix()
PLOT_PATH = Path(BASE_OUTPUT).joinpath("plot.png").as_posix()
