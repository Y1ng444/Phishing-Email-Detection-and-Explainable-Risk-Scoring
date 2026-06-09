"""
Test script to demonstrate multi-CSV dataset loading functionality.
Run this to verify the new auto-discovery feature works correctly.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocessing import load_dataset
from src.utils import setup_logger

logger = setup_logger(__name__)


def test_load_dataset():
    """Test the multi-CSV loading functionality."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Multi-CSV Auto-Discovery Dataset Loader")
    logger.info("=" * 80 + "\n")

    try:
        # Load dataset (automatically discovers all CSVs in data/ directory)
        df, total_rows = load_dataset()

        logger.info(f"\nTest Result: SUCCESS")
        logger.info(f"  Loaded dataframe shape: {df.shape}")
        logger.info(f"  Total rows before cleaning: {total_rows:,}")
        logger.info(f"  Total rows after cleaning: {df.shape[0]:,}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Data types: {df.dtypes.to_dict()}")

        return True

    except Exception as e:
        logger.error(f"\nTest Result: FAILED")
        logger.error(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_load_dataset()
    sys.exit(0 if success else 1)
