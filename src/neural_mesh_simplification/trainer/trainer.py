import logging
import os
from multiprocessing import Event, Process
from typing import Dict, Any

import torch
from dgl.dataloading import GraphDataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

from .resource_monitor import monitor_resources
from ..data import MeshSimplificationDataset
from ..data.dataset import collate, dgl_to_trimesh
from ..losses import CombinedMeshSimplificationLoss
from ..metrics import chamfer_distance, normal_consistency, edge_preservation, hausdorff_distance
from ..models import NeuralMeshSimplification

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Initializing trainer...")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        logger.info("Initializing model...")
        self.model = NeuralMeshSimplification(
            input_dim=config["model"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            edge_hidden_dim=config["model"]["edge_hidden_dim"],
            num_layers=config["model"]["num_layers"],
            k=config["model"]["k"],
            edge_k=config["model"]["edge_k"],
            target_ratio=config["model"]["target_ratio"],
        ).to(self.device)

        logger.debug("Setting up optimizer and loss...")
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
        self.criterion = CombinedMeshSimplificationLoss(
            lambda_c=config["loss"]["lambda_c"],
            lambda_e=config["loss"]["lambda_e"],
            lambda_o=config["loss"]["lambda_o"],
        ).to(self.device)

        self.early_stopping_patience = config["training"]["early_stopping_patience"]
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logger.debug("Preparing data loaders...")
        self.train_loader, self.val_loader = self._prepare_data_loaders()
        logger.info("Trainer initialization complete.")

        if "monitor_resources" in config:
            logger.info("Monitoring resource usage")
            self.monitor_resources = True
            self.stop_event = Event()
            self.monitor_process = None
        else:
            self.monitor_resources = False

    def _prepare_data_loaders(self):
        logger.info(f"Loading dataset from {self.config['data']['data_dir']}")
        dataset = MeshSimplificationDataset(
            data_dir=self.config["data"]["data_dir"],
            preprocess=False  # Can be False, because data has been prepared before training
        )
        logger.debug(f"Dataset size: {len(dataset)}")

        val_size = int(len(dataset) * self.config["data"]["val_split"])
        train_size = len(dataset) - val_size
        logger.info(f"Splitting dataset: {train_size} train, {val_size} validation")

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        assert len(val_dataset) > 0, \
            f"There is not enough data to define an evaluation set. len(dataset)={len(dataset)}, train_size={train_size}, val_size={val_size}"

        num_workers = self.config["training"].get("num_workers", os.cpu_count())
        logger.info(f"Using {num_workers} workers for data loading")

        train_loader = GraphDataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate
        )

        val_loader = GraphDataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate
        )
        logger.info("Data loaders prepared successfully")

        return train_loader, val_loader

    def train(self):
        if self.monitor_resources:
            main_pid = os.getpid()
            self.monitor_process = Process(target=monitor_resources, args=(self.stop_event, main_pid))
            self.monitor_process.start()

        logging.debug("Training started")

        try:
            for epoch in range(self.config["training"]["num_epochs"]):
                loss = self._train_one_epoch(epoch)

                logging.info(
                    f"Epoch [{epoch + 1}/{self.config['training']['num_epochs']}], Loss: {loss / len(self.train_loader)}"
                )

                val_loss = self._validate()
                logging.info(f"Epoch [{epoch + 1}/{self.config['training']['num_epochs']}], Validation Loss: {val_loss}")

                self.scheduler.step(val_loss)

                # Save the checkpoint
                self._save_checkpoint(epoch, val_loss)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Early stop as required
                if self._early_stopping(val_loss):
                    logging.info("Early stopping triggered.")
                    break

        except Exception as e:
            logger.error(f"{str(e)}")
        finally:
            if self.monitor_resources:
                self.stop_event.set()
                self.monitor_process.join()
                print()  # New line after monitoring output

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        logger.debug(f"Starting epoch {epoch + 1}")

        for batch_idx, batch in enumerate(self.train_loader):
            logger.debug(f"Processing batch {batch_idx + 1}")
            for orig_graph, orig_faces in zip(*batch):
                self.optimizer.zero_grad()

                orig_graph = orig_graph.to(self.device)
                s_graph, s_faces, face_probs = self.model(orig_graph)

                loss = self.criterion(orig_graph, orig_faces, s_graph, s_faces, face_probs)

                del orig_graph
                del orig_faces
                del s_graph
                del s_faces
                del face_probs

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _validate(self) -> float:
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                for orig_graph, orig_faces in zip(*batch):
                    s_graph, s_faces, face_probs = self.model(orig_graph)
                    loss = self.criterion(orig_graph, orig_faces, s_graph, s_faces, face_probs)

                    del orig_graph
                    del orig_faces
                    del s_graph
                    del s_faces
                    del face_probs

                    val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def _save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")

        logging.debug(f"Saving checkpoint to {checkpoint_path}")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path,
        )
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            logging.debug(f"Saving best model to {best_model_path}")
            torch.save(self.model.state_dict(), best_model_path)

            # Remove old checkpoints to save space
            for old_checkpoint in os.listdir(self.checkpoint_dir):
                if old_checkpoint.startswith("checkpoint_") and old_checkpoint != f"checkpoint_epoch_{epoch + 1}.pth":
                    os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))

    def _early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        return self.early_stopping_counter >= self.early_stopping_patience

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["val_loss"]
        logging.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")

    def log_metrics(self, metrics: Dict[str, float], epoch: int):
        log_message = f"Epoch [{epoch + 1}/{self.config['training']['num_epochs']}], "
        log_message += ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        logging.info(log_message)

    def evaluate(self, data_loader: GraphDataLoader) -> Dict[str, float]:
        self.model.eval()
        metrics = {
            "chamfer_distance": 0.0,
            "normal_consistency": 0.0,
            "edge_preservation": 0.0,
            "hausdorff_distance": 0.0
        }
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                for orig_graph, orig_faces in zip(*batch):
                    s_graph, s_faces, face_probs = self.model(orig_graph)

                    orig_mesh = dgl_to_trimesh(orig_graph, orig_faces)
                    s_mesh = dgl_to_trimesh(s_graph, s_faces)

                    metrics["chamfer_distance"] += chamfer_distance(orig_mesh, s_mesh)
                    metrics["normal_consistency"] += normal_consistency(orig_mesh)
                    metrics["edge_preservation"] += edge_preservation(orig_mesh, s_mesh)
                    metrics["hausdorff_distance"] += hausdorff_distance(orig_mesh, s_mesh)

        for key in metrics:
            metrics[key] /= len(data_loader)

        return metrics

    def handle_error(self, error: Exception):
        logging.error(f"An error occurred: {error}")
        if isinstance(error, RuntimeError) and "out of memory" in str(error):
            logging.error("Out of memory error. Attempting to recover.")
            torch.cuda.empty_cache()
        else:
            raise error

    def save_training_state(self, state_path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "early_stopping_counter": self.early_stopping_counter,
            },
            state_path,
        )

    def load_training_state(self, state_path: str):
        state = torch.load(state_path)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.best_val_loss = state["best_val_loss"]
        self.early_stopping_counter = state["early_stopping_counter"]
        logging.info(f"Loaded training state from {state_path}")
