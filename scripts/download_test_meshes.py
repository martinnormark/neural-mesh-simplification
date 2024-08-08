import argparse
import os
import shutil
from huggingface_hub import snapshot_download

# abc_train is really large (+5k meshes)
global folder_patterns
folder_patterns = ["abc_extra_noisy/03_meshes/*.ply", "abc_train/03_meshes/*.ply"]


def download_meshes(target_folder):
    wip_folder = os.path.join(target_folder, "wip")
    os.makedirs(wip_folder, exist_ok=True)

    snapshot_download(
        repo_id="perler/ppsurf",
        repo_type="dataset",
        cache_dir=wip_folder,
        allow_patterns=folder_patterns[0],
    )

    # Move files from wip folder to target folder
    for root, _, files in os.walk(wip_folder):
        for file in files:
            if file.endswith(".ply"):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(target_folder, file)
                shutil.copy2(src_file, dest_file)
                os.remove(src_file)

    # Remove the wip folder
    shutil.rmtree(wip_folder)


def main():
    parser = argparse.ArgumentParser(
        description="Download test meshes from Hugging Face Hub."
    )
    parser.add_argument(
        "--target-folder",
        type=str,
        required=True,
        help="The target folder path where the meshes will be downloaded.",
    )
    args = parser.parse_args()
    download_meshes(args.target_folder)


if __name__ == "__main__":
    main()
