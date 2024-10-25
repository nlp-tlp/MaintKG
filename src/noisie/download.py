"""Script to download model checkpoints from Google Drive."""

import subprocess
import sys
from pathlib import Path
from typing import NoReturn, Optional

from loguru import logger


class CheckpointDownloaderError(Exception):
    """Base exception class for checkpoint downloader errors."""

    pass


class DependencyInstallError(CheckpointDownloaderError):
    """Raised when there's an error installing dependencies."""

    pass


def install_gdown() -> None:
    """
    Install the gdown package using pip.

    Raises:
        DependencyInstallError: If gdown installation fails.
    """
    try:
        logger.info("Installing gdown package...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info("Successfully installed gdown")
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Failed to install gdown: {e.stderr.decode() if e.stderr else str(e)}"
        )
        logger.error(error_msg)
        raise DependencyInstallError(error_msg) from e


def ensure_gdown() -> None:
    """
    Ensure gdown is available, installing it if necessary.

    Raises:
        DependencyInstallError: If gdown installation fails.
        ImportError: If gdown cannot be imported after installation.
    """
    try:
        import gdown

        logger.debug("gdown is already installed")
    except ImportError:
        install_gdown()
        try:
            import gdown
        except ImportError as e:
            error_msg = "Failed to import gdown even after installation"
            logger.error(error_msg)
            raise ImportError(error_msg) from e


def create_target_directory(directory: Path) -> None:
    """
    Create the target directory if it doesn't exist.

    Args:
        directory: Path to the directory to create.

    Raises:
        OSError: If directory creation fails.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured target directory exists: {directory}")
    except OSError as e:
        error_msg = f"Failed to create directory {directory}: {str(e)}"
        logger.error(error_msg)
        raise OSError(error_msg) from e


def download_checkpoints(url: str, target_dir: Path) -> None:
    """
    Download model checkpoints from Google Drive.

    Args:
        url: Google Drive folder URL containing the checkpoints.
        target_dir: Local directory path where checkpoints will be saved.

    Raises:
        CheckpointDownloaderError: If the download fails.
    """
    import gdown  # Import here after ensuring it's installed

    try:
        logger.info(f"Downloading checkpoints from {url} to {target_dir}")
        result = gdown.download_folder(url=url, output=str(target_dir), quiet=False)
        if not result:
            raise CheckpointDownloaderError("Download failed - gdown returned False")
        logger.info("Successfully downloaded checkpoints")
    except Exception as e:
        error_msg = f"Failed to download checkpoints: {str(e)}"
        logger.error(error_msg)
        raise CheckpointDownloaderError(error_msg) from e


def main() -> Optional[NoReturn]:
    """
    Function that orchestrate checkpoint downloading.

    This function:
    1. Ensures gdown is installed
    2. Creates the target directory
    3. Downloads the checkpoints
    4. Handles user interaction

    Returns:
        None, or exits with error code 1 on failure.
    """
    try:
        # Initialize
        ensure_gdown()

        # Setup paths
        file_dir = Path(__file__).parent
        target_dir = file_dir / "lightning_logs" / "version_22_512_final_maintnormie"
        create_target_directory(target_dir)

        # Download checkpoints
        url = "https://drive.google.com/drive/folders/1if6dkFoVOdhdpZSWuJ38_o3zrGkSNWgv"
        download_checkpoints(url, target_dir)

        logger.info("All operations completed successfully")

    except (
        DependencyInstallError,
        ImportError,
        OSError,
        CheckpointDownloaderError,
    ) as e:
        logger.error(f"Fatal error: {str(e)}")
        if not sys.flags.interactive:
            input("\nPress Enter to exit...")
            sys.exit(1)
        return

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        if not sys.flags.interactive:
            sys.exit(1)
        return

    if not sys.flags.interactive:
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
