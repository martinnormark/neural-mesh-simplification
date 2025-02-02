import logging
import os
import time
import psutil
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from neural_mesh_simplification.data.dataset import MeshSimplificationDataset
from tqdm import tqdm
from tabulate import tabulate


class DataLoaderProfiler:
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        shuffle: bool = False,
        follow_batch_fields=None,
        log_interval: int = 10,
    ):
        """
        Initialize the profiler with dataset parameters.

        Args:
            data_dir (str): Path to the dataset directory.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of worker processes.
            shuffle (bool): Whether to shuffle the dataset.
            follow_batch_fields (list, optional): List of batch attributes to follow.
            log_interval (int): Frequency of logging progress (in batches).
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.follow_batch_fields = follow_batch_fields or ["x", "pos"]
        self.log_interval = log_interval

        self.dataset = MeshSimplificationDataset(data_dir=data_dir)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            follow_batch=self.follow_batch_fields,
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Dataset loaded from {data_dir} with {len(self.dataset)} samples."
        )

    def compute_batch_memory_usage(self, batch) -> float:
        """
        Compute the memory usage in MB for a given batch.

        Args:
            batch: A batch from the data loader.

        Returns:
            float: Memory usage in megabytes.
        """
        memory_bytes = 0
        for key in batch.keys():
            value = getattr(batch, key)
            if torch.is_tensor(value):
                memory_bytes += value.element_size() * value.nelement()
        return memory_bytes / (1024 * 1024)

    def profile(self) -> dict:
        """
        Profile the data loader over all batches and return summary statistics.

        Returns:
            dict: A dictionary containing batch statistics and memory usage metrics.
        """
        batch_times = []
        batch_memory = []
        vertices_per_batch = []
        faces_per_batch = []
        edges_per_batch = []

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        start_time = time.perf_counter()

        for batch_idx, batch in enumerate(
            tqdm(self.dataloader, desc="Profiling batches", leave=False)
        ):
            batch_start = time.perf_counter()

            num_vertices = getattr(batch, "num_nodes", 0)
            num_faces = (
                batch.face.shape[1]
                if hasattr(batch, "face")
                and batch.face is not None
                and batch.face.numel() > 0
                else 0
            )
            num_edges = (
                batch.edge_index.shape[1]
                if hasattr(batch, "edge_index")
                and batch.edge_index is not None
                and batch.edge_index.numel() > 0
                else 0
            )

            batch_memory.append(self.compute_batch_memory_usage(batch))
            vertices_per_batch.append(num_vertices)
            faces_per_batch.append(num_faces)
            edges_per_batch.append(num_edges)

            batch_times.append(time.perf_counter() - batch_start)

        total_time = time.perf_counter() - start_time
        final_memory = process.memory_info().rss / (1024 * 1024)

        summary = {
            "total_time": total_time,
            "avg_batch_time": np.mean(batch_times) if batch_times else 0,
            "memory_increase": final_memory - initial_memory,
            "avg_vertices": np.mean(vertices_per_batch) if vertices_per_batch else 0,
            "avg_faces": np.mean(faces_per_batch) if faces_per_batch else 0,
            "avg_edges": np.mean(edges_per_batch) if edges_per_batch else 0,
            "avg_memory": np.mean(batch_memory) if batch_memory else 0,
            "min_batch_memory": np.min(batch_memory) if batch_memory else 0,
            "max_batch_memory": np.max(batch_memory) if batch_memory else 0,
            "vertices_range": (
                np.min(vertices_per_batch) if vertices_per_batch else 0,
                np.max(vertices_per_batch) if vertices_per_batch else 0,
            ),
            "faces_range": (
                np.min(faces_per_batch) if faces_per_batch else 0,
                np.max(faces_per_batch) if faces_per_batch else 0,
            ),
            "edges_range": (
                np.min(edges_per_batch) if edges_per_batch else 0,
                np.max(edges_per_batch) if edges_per_batch else 0,
            ),
        }
        return summary

    def log_summary(self, summary: dict) -> None:
        """
        Log the summary statistics in a tabulated format.

        Args:
            summary (dict): Dictionary containing the profiling results.
        """
        self.logger.info("\n\n========= Dataset Profiling Results =========\n")

        # Timing and Memory Overview
        timing_memory_data = [
            ["Total Processing Time", f"{summary['total_time']:.2f} s"],
            ["Average Batch Time", f"{summary['avg_batch_time']:.4f} s"],
            ["Memory Usage Increase", f"{summary['memory_increase']:.2f} MB"],
        ]
        self.logger.info("\nTiming and Memory Overview:")
        self.logger.info(
            "\n"
            + tabulate(
                timing_memory_data,
                tablefmt="psql",
                colalign=("left", "right"),
                maxcolwidths=[25, 15],
            )
            + "\n"
        )

        # Batch Statistics
        batch_stats_data = [
            ["Metric", "Average", "Min", "Max"],
            [
                "Vertices",
                f"{summary['avg_vertices']:,.1f}",
                f"{summary['vertices_range'][0]:,d}",
                f"{summary['vertices_range'][1]:,d}",
            ],
            [
                "Faces",
                f"{summary['avg_faces']:,.1f}",
                f"{summary['faces_range'][0]:,d}",
                f"{summary['faces_range'][1]:,d}",
            ],
            [
                "Edges",
                f"{summary['avg_edges']:,.1f}",
                f"{summary['edges_range'][0]:,d}",
                f"{summary['edges_range'][1]:,d}",
            ],
            [
                "Memory (MB)",
                f"{summary['avg_memory']:.2f}",
                f"{summary['min_batch_memory']:.2f}",
                f"{summary['max_batch_memory']:.2f}",
            ],
        ]
        self.logger.info("\nBatch Statistics:")
        self.logger.info(
            "\n"
            + tabulate(
                batch_stats_data,
                headers="firstrow",
                tablefmt="psql",
                colalign=("left", "right", "right", "right"),
                maxcolwidths=[15, 12, 12, 12],
            )
            + "\n"
        )

        # Add dataset info with proper string formatting
        dataset_info = [
            ["Configuration", "Value"],  # Add header row
            ["Total Samples", f"{len(self.dataset):,d}"],
            ["Batch Size", f"{self.batch_size}"],
            ["Number of Workers", f"{self.num_workers}"],
            [
                "Shuffle Enabled",
                "Yes" if self.shuffle else "No",
            ],  # Convert boolean to Yes/No
        ]
        self.logger.info("\nDataset Configuration:")
        self.logger.info(
            "\n"
            + tabulate(
                dataset_info,
                headers="firstrow",
                tablefmt="psql",
                colalign=("left", "right"),
                maxcolwidths=[20, 10],
            )
            + "\n"
        )
