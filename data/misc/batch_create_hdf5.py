import os
import math
import h5py
import argparse
import numpy as np
from glob import glob
from PIL import Image

ORIGINAL_FRONT_HEIGHT = 1280
ORIGINAL_FRONT_WIDTH = 1920
ORIGINAL_BACK_HEIGHT = 886
ORIGINAL_BACK_WIDTH = 1920

FRONT_CAMERA_IDS = ["1", "2", "3"]
BACK_CAMERA_IDS = ["4", "5"]


def get_video_paths(args):
    video_paths = []
    if args.subset:
        video_paths += sorted(
            glob(os.path.join(args.data_root, "training", "segment-*"))
        )[:8]
        video_paths += sorted(
            glob(os.path.join(args.data_root, "validation", "segment-*"))
        )[:2]
    else:
        video_paths += sorted(
            glob(os.path.join(args.data_root, "training", "segment-*"))
        )
        video_paths += sorted(
            glob(os.path.join(args.data_root, "validation", "segment-*"))
        )

    return video_paths


def batch_run(args):
    os.makedirs(args.store_dir, exist_ok=True)
    os.makedirs(args.slurm_dir, exist_ok=True)

    video_paths = get_video_paths(args)

    job_array = list(range(len(video_paths)))
    job_array = list(range(math.ceil(len(video_paths) / args.process_num)))

    job_name = f"waymo_hdf5_0-{len(job_array)-1}"
    job_command = (
        f"python batch_create_hdf5.py "
        f"--data_root {args.data_root} "
        f"--store_dir {args.store_dir} "
        f"--process_num {args.process_num} "
        f"--mode single"
    )
    if args.subset:
        job_command += " --subset"

    slurm_file = os.path.join(args.slurm_dir, job_name + ".slurm")
    with open(slurm_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=%s\n" % job_name)
        f.write("#SBATCH --cpus-per-task=4\n")
        f.write("#SBATCH --time=8:00:00\n")
        f.write("#SBATCH --array=%s\n" % ",".join([str(c) for c in job_array]))
        f.write("#SBATCH --nodes=%d\n" % 1)
        f.write(
            "#SBATCH --output=%s\n"
            % os.path.join(args.slurm_dir, job_name + ".out")
        )
        f.write(
            "#SBATCH --error=%s\n"
            % os.path.join(args.slurm_dir, job_name + ".err")
        )

        f.write(job_command + "\n")

    s = "sbatch %s" % slurm_file
    os.system(s)


def single_run(args):
    jobid = os.getenv("SLURM_ARRAY_TASK_ID")

    video_paths = get_video_paths(args)

    if jobid is not None:
        jobid = int(jobid)
        video_paths = video_paths[
            jobid * args.process_num : (jobid + 1) * args.process_num
        ]

    front_width_ratio = args.front_img_width / ORIGINAL_FRONT_WIDTH
    front_height_ratio = args.front_img_height / ORIGINAL_FRONT_HEIGHT
    front_ratio = np.array(
        [
            front_width_ratio,
            front_height_ratio,
            front_width_ratio,
            front_height_ratio,
        ]
    )
    back_width_ratio = args.back_img_width / ORIGINAL_BACK_WIDTH
    back_height_ratio = args.back_img_height / ORIGINAL_BACK_HEIGHT
    back_ratio = np.array(
        [
            back_width_ratio,
            back_height_ratio,
            back_width_ratio,
            back_height_ratio,
        ]
    )

    for video_path in video_paths:
        print(video_path)
        split = os.path.split(os.path.split(video_path)[0])[1]
        os.makedirs(os.path.join(args.store_dir, split), exist_ok=True)
        frame_paths = sorted(glob(os.path.join(video_path, "*")))

        hdf5_path = os.path.join(
            args.store_dir, split, f"{os.path.basename(video_path)}.hdf5"
        )
        if os.path.exists(hdf5_path):
            continue
        f = h5py.File(hdf5_path, "w")

        # Intrinsic matrix and extrinsic matrix should be
        # all the same for all images in the same sequence
        f.create_group("intrinsic")
        f.create_group("extrinsic")
        for camera in range(1, 6):
            # Load intrinsic
            intrinsic = np.load(
                os.path.join(frame_paths[0], f"{camera}_intrinsic.npy")
            )
            f["intrinsic"].create_dataset(str(camera), data=intrinsic)

            # Load extrinsic
            extrinsic = np.load(
                os.path.join(frame_paths[0], f"{camera}_extrinsic.npy")
            )
            f["extrinsic"].create_dataset(str(camera), data=extrinsic)

        for i, frame_path in enumerate(frame_paths):
            frame_name = str(i)
            f.create_group(frame_name)

            # Load PTP
            ptp = np.load(os.path.join(frame_path, "pose.npy"))
            f[frame_name].create_dataset("PTP", data=ptp)

            for camera in range(1, 6):
                camera = str(camera)
                f[frame_name].create_group(camera)

                # Load image
                img = Image.open(os.path.join(frame_path, f"{camera}.png"))
                if camera in FRONT_CAMERA_IDS:
                    img = img.resize(
                        (args.front_img_width, args.front_img_height)
                    )
                elif camera in BACK_CAMERA_IDS:
                    img = img.resize(
                        (args.back_img_width, args.back_img_height)
                    )
                f[frame_name][camera].create_dataset("image", data=img)

                # Load id
                id_arr = np.load(os.path.join(frame_path, f"{camera}_id.npy"))
                id_arr = [
                    n.encode("ascii")
                    for n in np.load(
                        os.path.join(frame_path, f"{camera}_id.npy")
                    )
                ]
                f[frame_name][camera].create_dataset("id", data=id_arr)

                # Load label
                lbl_arr = np.load(
                    os.path.join(frame_path, f"{camera}_label.npy")
                )
                if len(lbl_arr) > 0:
                    if camera in FRONT_CAMERA_IDS:
                        lbl_arr *= front_ratio
                    elif camera in BACK_CAMERA_IDS:
                        lbl_arr *= back_ratio
                f[frame_name][camera].create_dataset("label", data=lbl_arr)

                # Load type
                type_arr = np.load(
                    os.path.join(frame_path, f"{camera}_type.npy")
                )
                f[frame_name][camera].create_dataset("type", data=type_arr)

        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", default="/scratch/jz3224/WaymoDataset", type=str
    )
    parser.add_argument("--store_dir", type=str)
    parser.add_argument(
        "--slurm_dir",
        default=".",
        type=str,
        help="where to store slrum script",
    )
    parser.add_argument(
        "--process_num",
        default=10,
        type=int,
        help="number of videos to process in each job",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="whether to process only subset or full dataset",
    )
    parser.add_argument("--front_img_height", default=320, type=int)
    parser.add_argument("--front_img_width", default=480, type=int)
    parser.add_argument("--back_img_height", default=222, type=int)
    parser.add_argument("--back_img_width", default=480, type=int)
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="batch or single",
    )
    args = parser.parse_args()

    if args.mode == "batch":
        batch_run(args)
    elif args.mode == "single":
        single_run(args)
