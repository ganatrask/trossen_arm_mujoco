"""
HuggingFace Hub uploader with retry logic and batch support.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Set

from huggingface_hub import HfApi, create_repo, list_repo_files, upload_folder


class HuggingFaceUploader:
    """
    Handles uploading datasets to HuggingFace Hub with retry logic.

    Features:
    - Automatic repo creation
    - Batch upload with exponential backoff retry
    - Resume support (tracks uploaded files)
    - README/dataset card updates
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        dry_run: bool = False,
    ):
        """
        Initialize the uploader.

        Args:
            repo_id: HuggingFace repo ID (e.g., "username/dataset-name")
            token: HuggingFace API token (uses cached token if None)
            private: Whether to create a private repository
            dry_run: If True, don't actually upload (for testing)
        """
        self.repo_id = repo_id
        self.token = token
        self.private = private
        self.dry_run = dry_run
        self.api = HfApi(token=token)
        self._uploaded_files_cache: Optional[Set[str]] = None

    def ensure_repo_exists(self) -> bool:
        """
        Ensure the dataset repository exists, creating it if necessary.

        Returns:
            True if repo exists or was created successfully
        """
        if self.dry_run:
            print(f"[DRY RUN] Would ensure repo exists: {self.repo_id}")
            return True

        try:
            create_repo(
                repo_id=self.repo_id,
                token=self.token,
                repo_type="dataset",
                private=self.private,
                exist_ok=True,
            )
            return True
        except Exception as e:
            print(f"Error creating/checking repo: {e}")
            return False

    def upload_batch(
        self,
        local_dir: str,
        path_in_repo: str,
        max_retries: int = 3,
        ignore_patterns: Optional[List[str]] = None,
    ) -> bool:
        """
        Upload a batch directory to HuggingFace with retry logic.

        Args:
            local_dir: Local directory containing files to upload
            path_in_repo: Path in the repository (e.g., "data/batch_000")
            max_retries: Maximum number of retry attempts
            ignore_patterns: Patterns to ignore during upload

        Returns:
            True if upload succeeded, False otherwise
        """
        if self.dry_run:
            files = list(Path(local_dir).glob("*"))
            print(f"[DRY RUN] Would upload {len(files)} files from {local_dir} to {path_in_repo}")
            return True

        if not os.path.exists(local_dir):
            print(f"Local directory does not exist: {local_dir}")
            return False

        for attempt in range(max_retries):
            try:
                upload_folder(
                    folder_path=local_dir,
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token,
                    ignore_patterns=ignore_patterns,
                )
                # Invalidate cache after successful upload
                self._uploaded_files_cache = None
                return True
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"Upload attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        return False

    def upload_file(
        self,
        local_path: str,
        path_in_repo: str,
        max_retries: int = 3,
    ) -> bool:
        """
        Upload a single file to HuggingFace with retry logic.

        Args:
            local_path: Path to local file
            path_in_repo: Path in the repository
            max_retries: Maximum number of retry attempts

        Returns:
            True if upload succeeded, False otherwise
        """
        if self.dry_run:
            print(f"[DRY RUN] Would upload {local_path} to {path_in_repo}")
            return True

        if not os.path.exists(local_path):
            print(f"Local file does not exist: {local_path}")
            return False

        for attempt in range(max_retries):
            try:
                self.api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token,
                )
                # Invalidate cache after successful upload
                self._uploaded_files_cache = None
                return True
            except Exception as e:
                wait_time = 2 ** attempt
                print(f"Upload attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        return False

    def update_readme(self, content: str, max_retries: int = 3) -> bool:
        """
        Update the repository README.md with dataset card content.

        Args:
            content: Markdown content for the README
            max_retries: Maximum number of retry attempts

        Returns:
            True if update succeeded, False otherwise
        """
        if self.dry_run:
            print(f"[DRY RUN] Would update README.md ({len(content)} chars)")
            return True

        for attempt in range(max_retries):
            try:
                self.api.upload_file(
                    path_or_fileobj=content.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token,
                )
                return True
            except Exception as e:
                wait_time = 2 ** attempt
                print(f"README update attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        return False

    def get_uploaded_files(self, refresh: bool = False) -> Set[str]:
        """
        Get list of files already uploaded to the repository.

        Useful for resume functionality - skip files that already exist.

        Args:
            refresh: Force refresh the cache

        Returns:
            Set of file paths in the repository
        """
        if self.dry_run:
            return set()

        if self._uploaded_files_cache is not None and not refresh:
            return self._uploaded_files_cache

        try:
            files = list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
            )
            self._uploaded_files_cache = set(files)
            return self._uploaded_files_cache
        except Exception as e:
            print(f"Error listing repo files: {e}")
            return set()

    def is_batch_uploaded(self, batch_num: int) -> bool:
        """
        Check if a batch has been uploaded.

        Args:
            batch_num: Batch number to check

        Returns:
            True if at least one file from the batch exists in repo
        """
        if self.dry_run:
            return False

        uploaded = self.get_uploaded_files()
        batch_prefix = f"data/batch_{batch_num:03d}/"
        return any(f.startswith(batch_prefix) for f in uploaded)
