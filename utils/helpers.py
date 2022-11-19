import os
import zipfile

import cv2
import numpy as np
import torchvision.transforms as transforms


def unzip_input_file(file_name) -> None:
    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall()


# Reading .npy files
def load_data(file_path: str):
    return np.load(file_path, allow_pickle=True)


def resize_np_array(arr, resize_shape):
    for i in range(len(arr)):
        arr[i][0] = cv2.resize(arr[i][0], (resize_shape, resize_shape))
    return arr


def fix_test_data(test_data):
    for i in range(len(test_data)):
        test_data[i][1] = np.array(test_data[i][1])
    return test_data


def prepare_train_valid_test(
    BASE_DIR, folder_name, train_name, test_name, IMG_SIZE_ALEXNET, train_size=4800
):
    train_data = load_data(os.path.join(BASE_DIR, folder_name, train_name))
    test_data = load_data(os.path.join(BASE_DIR, folder_name, test_name))

    # fix the test_data by converting list to numpy array
    test_data = fix_test_data(test_data)

    # resize all_images in npy files

    # In order to implement ALEXNET, we are resizing them to (227,227,3)

    train_data = resize_np_array(train_data, IMG_SIZE_ALEXNET)
    test_data = resize_np_array(test_data, IMG_SIZE_ALEXNET)

    train = train_data[:train_size]
    cv = train_data[train_size:]
    return train, cv, test_data

def create_transform():
    # transform for input image
    transform = transforms.Compose(
        transforms=[
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform