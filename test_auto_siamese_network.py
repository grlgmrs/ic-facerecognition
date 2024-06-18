from utils import (
    load_images_from_folder,
    label_and_shuffle_pairs,
    create_test_pairs,
    load_model,
    plot_roc,
)

base_url = "/var/www/facul/ic/__old__/images/train"
start_index, end_index = 190000, 200000
percentage_correct_pairs = 0.5

sharp_images = load_images_from_folder(f"{base_url}/sharp", start_index, end_index)
blur_images = load_images_from_folder(f"{base_url}/blur", start_index, end_index)

correct_pairs, incorrect_pairs = create_test_pairs(
    sharp_images, blur_images, percentage_correct_pairs
)
pairs, labels = label_and_shuffle_pairs(correct_pairs, incorrect_pairs)

model = load_model()

pairs = [pairs[:, 0], pairs[:, 1]]


folder_to_save = "resnet50v2_output_100epochs"
plot_roc(model, pairs, labels, folder_to_save)
