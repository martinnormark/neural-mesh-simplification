import argparse

from neural_mesh_simplification.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Neural Mesh Simplification model.")
    parser.add_argument("--eval-data-path", type=str, required=True, help="Path to the evaluation data directory.")
    parser.add_argument("--config", type=str, required=False, help="Path to the evaluation configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    return parser.parse_args()


def load_config(config_path):
    import yaml
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    config["data"]["data_dir"] = args.eval_data_path

    trainer = Trainer(config)
    trainer.load_checkpoint(args.checkpoint)

    evaluation_metrics = trainer.evaluate(trainer.val_loader)
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
