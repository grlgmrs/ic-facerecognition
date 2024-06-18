from utils import (
    load_pairs_and_labels,
    normalize_pairs,
    plot_training,
    save_model,
    load_model,
)
import config

train_range = 200

start_index = 0
end_index = start_index + train_range

(pairs, labels) = load_pairs_and_labels(start_index, end_index)
pairs = normalize_pairs(pairs)

model = load_model()

# shape (num_pairs, 2, 218, 117, 3)

history = model.fit(
    [pairs[:, 0], pairs[:, 1]],
    labels,
    validation_split=0.2,
    batch_size=config.BATCH_SIZE,
    epochs=15,
)


save_model(model)

plot_training(history, config.PLOT_PATH)
