# ruff: noqa: E501
"""Auto-sklearn classifier."""

import json
import os

from aiod.models.apis import _ModelPkgSklearnEstimator

_SCIKIT_LEARN_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "scikit_learn_data.json"
)


def _load_sklearn_data():
    if not os.path.exists(_SCIKIT_LEARN_DATA_PATH):
        return {"_obj_dict": {}, "_type_of_objs": {}, "_objs_by_type": {}}
    with open(_SCIKIT_LEARN_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_sklearn_data = _load_sklearn_data()


class AiodPkg__Sklearn(_ModelPkgSklearnEstimator):
    _tags = {
        "pkg_id": "__multiple",
        "python_dependencies": "scikit-learn",
        "pkg_pypi_name": "scikit-learn",
        "object_types": ["classifier", "regressor"],
    }

    _obj_dict = _sklearn_data.get("_obj_dict", {})
    _type_of_objs = _sklearn_data.get("_type_of_objs", {})
    _objs_by_type = _sklearn_data.get("_objs_by_type", {})
