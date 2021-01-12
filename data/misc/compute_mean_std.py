import os
import h5py
import argparse
import numpy as np
import multiprocessing as mp


def compute(hdf5_path, mode, img_mean=None):
    if mode == "std":
        assert img_mean is not None

    print(hdf5_path)

    total = np.zeros(3)
    count = 0
    file = h5py.File(hdf5_path, "r")
    for file_key, frame in file.items():
        if file_key in ["intrinsic", "extrinsic"]:
            continue

        for frame_key, camera in frame.items():
            if frame_key == "PTP":
                continue

            img = camera["image"][:].reshape((-1, 3)) / 255.0

            if mode == "mean":
                total += np.sum(img, axis=0)
            if mode == "std":
                total += np.sum((img - img_mean) ** 2, axis=0)

            count += img.shape[0]

    file.close()

    return [total, count]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="/scratch/zz2332/WaymoDataset",
        help="Where to load the data",
    )
    args = parser.parse_args([])

    # Only use training images
    data_root = os.path.join(args.data_root, "training")

    # Gather paths
    paths = [os.path.join(data_root, path) for path in os.listdir(data_root)]

    with mp.Pool(16) as pool:
        mean_results = pool.starmap(
            compute, zip(paths, ["mean",] * len(paths))
        )
    pool.join()

    img_mean = sum([result[0] for result in mean_results]) / sum(
        [result[1] for result in mean_results]
    )

    with mp.Pool(16) as pool:
        std_results = pool.starmap(
            compute,
            zip(paths, ["std",] * len(paths), [img_mean,] * len(paths)),
        )
    pool.join()

    img_std = np.sqrt(
        sum([result[0] for result in std_results])
        / sum([result[1] for result in std_results])
    )

    print("mean:", img_mean)
    print("std:", img_std)
