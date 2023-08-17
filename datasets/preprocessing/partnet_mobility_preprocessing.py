import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fire import Fire
from loguru import logger
from natsort import natsorted

from datasets.preprocessing.base_preprocessing import BasePreprocessing


class PartNetMobilityPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/partnet-mobility-v0/dataset",
        save_dir: str = "./data/processed/partnet-mobility-v0",
        modes: tuple = ("train", "validation", "test"),
        n_jobs: int = -1,
        category: str = "Scissors",
        split_ratio: tuple = (0.8, 0.1, 0.1),
    ):
        assert len(split_ratio) == len(
            modes
        ), "split_ratio must have the same length as modes"

        if category is not None:
            save_dir = os.path.join(save_dir, category)

        super().__init__(data_dir, save_dir, modes, n_jobs)

        # Find all part names and model ids
        part_names = set()
        model_ids = []
        for model_dir in self.data_dir.iterdir():
            if not model_dir.is_dir():
                print(f"{model_dir} is not a directory")
                continue

            # Parse category
            meta_json_path = model_dir / "meta.json"
            with open(meta_json_path, "r") as f:
                meta_json = json.load(f)
            if meta_json["model_cat"] != category:
                continue

            model_id = model_dir.name
            model_ids.append(model_id)

            # Parse semantics
            with open(model_dir / "semantics.txt", "r") as f:
                lines = f.readlines()
            for line in lines:
                _, _, name = line.split()
                part_names.add(name)

        part_names = list(part_names)
        print(f"{category}:", part_names)
        self.part_names = part_names

        # Create label database
        label_database = {}
        for i, name in enumerate(part_names):
            color = np.uint8([11, 61, 127]) * (i + 1)  # rely on uint8 clipping
            label_database[i] = dict(name=name, color=color.tolist(), validation=True)
        self._save_yaml(self.save_dir / "label_database.yaml", label_database)

        # Generate splits
        model_ids = natsorted(model_ids)
        n = len(model_ids)
        ns = [0] + [int(n * r) for r in split_ratio]
        ns[-1] = n - sum(ns[:-1])
        ns = np.cumsum(ns)

        # Generate files (model ids)
        for i, mode in enumerate(self.modes):
            self.files[mode] = []
            for model_id in model_ids[ns[i] : ns[i + 1]]:
                self.files[mode].append(model_id)

    def process_file(self, model_id, mode):
        model_dir = self.data_dir / model_id

        # Parse semantics
        semantics = {}
        with open(model_dir / "semantics.txt", "r") as f:
            lines = f.readlines()
        for line in lines:
            link_name, _, link_semantic = line.split()
            semantics[link_name] = link_semantic
        link_names = list(semantics.keys())

        # result_json_path = model_dir / "result.json"
        # with open(result_json_path, "r") as f:
        #     result_json = json.load(f)
        # leaf_parts = get_leaf_parts(result_json)

        mobility_json_path = model_dir / "mobility_v2.json"
        with open(mobility_json_path, "r") as f:
            mobility_json = json.load(f)
        link_name_to_part_ids = {}
        part_ids_to_link_name = {}
        for link in mobility_json:
            link_id = link["id"]
            link_name = f"link_{link_id}"
            link_parts = link["parts"]
            part_ids = []
            for link_part in link_parts:
                assert "children" not in link_part or len(link_part["children"]) == 0, (model_id, link_part)
                part_id = link_part["id"]
                part_ids.append(part_id)
            link_name_to_part_ids[link_name] = part_ids
            for part_id in part_ids:
                assert part_id not in part_ids_to_link_name, "part id must be unique"
                part_ids_to_link_name[part_id] = link_name
                
        xyz_rgb = np.loadtxt(model_dir / "point_sample" / "pts-10000.pts")  # [N, 6]
        xyz = xyz_rgb[:, :3]
        rgb = (xyz_rgb[:, 3:] * 255).clip(0, 255)

        # dummy normals
        normals = np.zeros_like(xyz_rgb[:, :3])
        segment_ids = np.ones_like(xyz_rgb[:, 0:1])

        # mapping from part id to part name
        labels = np.loadtxt(model_dir / "point_sample" / "label-10000.txt", dtype=int)  # [N]
        seg_labels = []
        inst_labels = []
        for label in labels:
            link_name = part_ids_to_link_name[label]
            link_semantic = semantics[link_name]
            seg_label = self.part_names.index(link_semantic)
            seg_labels.append(seg_label)
            inst_label = link_names.index(link_name)
            inst_labels.append(inst_label)
        seg_labels = np.array(seg_labels)
        inst_labels = np.array(inst_labels)

        points = np.hstack(
            (xyz, rgb, normals, segment_ids, seg_labels[:, None], inst_labels[:, None])
        )
        gt_data = points[:, -2] * 1000 + points[:, -1] + 1

        filebase = {
            "filepath": model_id,
            "scene": model_id,
            "raw_filepath": str(model_id),
            "file_len": -1,
        }

        file_len = len(points)
        filebase["file_len"] = file_len

        processed_filepath = self.save_dir / mode / f"{model_id}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = self.save_dir / "instance_gt" / mode / f"{model_id}.txt"
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        filebase["color_mean"] = [
            float((points[:, 3] / 255).mean()),
            float((points[:, 4] / 255).mean()),
            float((points[:, 5] / 255).mean()),
        ]
        # NOTE: this is not the standard deviation, but the square sum
        filebase["color_std"] = [
            float(((points[:, 3] / 255) ** 2).mean()),
            float(((points[:, 4] / 255) ** 2).mean()),
            float(((points[:, 5] / 255) ** 2).mean()),
        ]
        return filebase

    def compute_color_mean_std(self, train_database_path: str):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean**2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)


def get_leaf_parts(part_annos: dict):
    """Get leaf parts from part annotations at any level."""
    leaf_parts = []
    for part_anno in part_annos:
        if "children" in part_anno:
            leaf_parts.extend(part_anno["children"])
        else:
            leaf_parts.append(part_anno)
    return leaf_parts


if __name__ == "__main__":
    Fire(PartNetMobilityPreprocessing)
