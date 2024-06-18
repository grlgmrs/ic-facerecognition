from build_siamese_network import build_siamese_network
from utils import load_pairs_and_labels, normalize_pairs, plot_training, save_model
import config

start_index, end_index = 0, 10000

(pairs, labels) = load_pairs_and_labels(start_index, end_index)
pairs = normalize_pairs(pairs)

model = build_siamese_network()

# shape (num_pairs, 2, 218, 117, 3)

history = model.fit(
    [pairs[:, 0], pairs[:, 1]],
    labels,
    validation_split=0.2,
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
)


save_model(model)

plot_training(history, config.PLOT_PATH)
