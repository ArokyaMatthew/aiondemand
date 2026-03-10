import argparse
import json
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class BaseLibraryUpdater(ABC):
    @abstractmethod
    def get_updates(self) -> Dict[str, Any]:
        """Return the dictionary of data to be serialized."""
        pass

    @abstractmethod
    def get_target_path(self) -> str:
        """Return the absolute path of the target JSON file."""
        pass

    def update(self, dry_run: bool = False):
        target_path = self.get_target_path()
        logger.info(f"Starting update for {self.__class__.__name__}")

        try:
            data = self.get_updates()
        except Exception as e:
            logger.error(
                f"Failed to generate updates for {self.__class__.__name__}: {e}"
            )
            return

        json_data = json.dumps(data, indent=4, sort_keys=True)

        if dry_run:
            logger.info(f"DRY RUN: Would write the following to {target_path}:\n")
            print(json_data)
            return

        # Write to a temporary file, then use atomic replace
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(target_path), suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json_data)
            os.replace(tmp_path, target_path)
            logger.info(f"Successfully updated {target_path}")
        except Exception as e:
            logger.error(f"Failed to write to {target_path}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class SklearnUpdater(BaseLibraryUpdater):
    def get_updates(self) -> Dict[str, Any]:
        try:
            import sklearn  # noqa
        except ImportError:
            raise RuntimeError(
                "scikit-learn is not installed. Will not update indices."
            )

        from aiod.utils._indexing._preindex_sklearn import (
            _all_sklearn_estimators_locdict,
            _generate_sklearn_objs_by_type,
            _generate_sklearn_types_of_obj,
        )

        obj_dict = _all_sklearn_estimators_locdict()
        type_of_objs = _generate_sklearn_types_of_obj()
        objs_by_type = _generate_sklearn_objs_by_type(type_of_objs)

        return {
            "_obj_dict": obj_dict,
            "_type_of_objs": type_of_objs,
            "_objs_by_type": objs_by_type,
        }

    def get_target_path(self) -> str:
        # Assuming script is run from project root, or specify absolute:
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "src", "aiod", "models", "sklearn_apis")
        )
        return os.path.join(base_dir, "scikit_learn_data.json")


def main():
    parser = argparse.ArgumentParser(
        description="Automate the updating of indexed libraries."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated dictionaries without writing to the files.",
    )
    args = parser.parse_args()

    updaters = [
        SklearnUpdater(),
    ]

    for updater in updaters:
        updater.update(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
