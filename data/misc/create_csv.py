import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root", default="/scratch/jz3224/WaymoDataset", type=str
)
parser.add_argument("--store_dir", type=str)
args = parser.parse_args()

count = 0


def f(video_path):
    print(video_path)
    results = []
    frame_paths = sorted(glob(os.path.join(video_path, "*")))
    for i, frame_path in enumerate(frame_paths):
        for camera in range(1, 6):
            result = {}

            result["episode_name"] = os.path.basename(video_path)

            # Add camera
            result["camera_name"] = camera

            # Add frame name
            result["frame_name"] = os.path.basename(frame_path)

            # Add info
            info = np.load(os.path.join(frame_path, f"{camera}_info.npy"))
            result["width"] = info[0]
            result["height"] = info[1]
            result["rolling_shutter_direction"] = info[2]
            result["frame_index"] = i

            results.append(result)

    return results


if __name__ == "__main__":
    os.makedirs(args.store_dir, exist_ok=True)

    for split in ["training", "validation"]:
        video_paths = sorted(
            glob(os.path.join(args.data_root, split, "segment-*"))
        )
        print(f"{split} {len(video_paths)}")
        with mp.Pool(16) as pool:
            results = pool.map(f, video_paths)
        pool.join()

        video_index = []
        for i, item in enumerate(results):
            video_index += [i,] * len(item)

        results = [item for sublist in results for item in sublist]

        column_names = [
            "episode_name",
            "frame_name",
            "camera_name",
            "width",
            "height",
            "rolling_shutter_direction",
            "frame_index",
        ]
        d = {name: [item[name] for item in results] for name in column_names}
        d["video_index"] = video_index
        df = pd.DataFrame(data=d)
        df.to_csv(os.path.join(args.store_dir, f"{split}.csv"))
