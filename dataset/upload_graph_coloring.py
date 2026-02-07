"""Upload extracted graph coloring data to HuggingFace.

Usage:
    python -m dataset.upload_graph_coloring \
        --repo_id ahn-ml/graph-coloring \
        --export_dir export \
        --configs '["graph-coloring-8", "graph-coloring-10"]'
"""
from typing import List
import os

from argdantic import ArgParser
from pydantic import BaseModel
from huggingface_hub import HfApi


cli = ArgParser()


class UploadConfig(BaseModel):
    repo_id: str = "ahn-ml/graph-coloring"
    export_dir: str = "export"
    configs: List[str] = ["graph-coloring-8", "graph-coloring-10"]
    create_repo: bool = True


@cli.command(singleton=True)
def upload(config: UploadConfig):
    api = HfApi()

    if config.create_repo:
        api.create_repo(repo_id=config.repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repo ready: {config.repo_id}")

    for cfg_name in config.configs:
        folder = os.path.join(config.export_dir, cfg_name)
        if not os.path.isdir(folder):
            print(f"[{cfg_name}] {folder} not found, skipping")
            continue

        print(f"[{cfg_name}] Uploading {folder} -> {cfg_name}/")
        api.upload_folder(
            folder_path=folder,
            path_in_repo=cfg_name,
            repo_id=config.repo_id,
            repo_type="dataset",
        )
        print(f"[{cfg_name}] Done")


if __name__ == "__main__":
    cli()
