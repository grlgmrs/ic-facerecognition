from utils import load_model, normalize_pairs, select_images
import numpy as np
import visualkeras

model = load_model()

visualkeras.layered_view(
    model, to_file="output.png", legend=True, draw_volume=False, scale_xy=1, scale_z=1
)  # write and show

# images = select_images(
#     "/var/www/facul/ic/__old__/images/train/mm",
#     "/var/www/facul/ic/__old__/images/train/mm",
# )

# sharp, blur = normalize_pairs(np.array(images))

# sharp = np.expand_dims(sharp, axis=0)
# blur = np.expand_dims(blur, axis=0)

# model = load_model()

# preds = model.predict([sharp, blur])
# prob = preds[0][0]

# print(f"Similarity: {prob:.2f}")
