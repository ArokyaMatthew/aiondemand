import json
import os
import pytest

from scripts.update_indices import SklearnUpdater


def test_sklearn_updater_generates_valid_dictionaries():
    """Test that the sklearn updater generates the expected keys and dictionaries."""
    
    updater = SklearnUpdater()
    updates = updater.get_updates()
    
    # Assert that all required dictionary keys are present
    assert "_obj_dict" in updates
    assert "_type_of_objs" in updates
    assert "_objs_by_type" in updates
    
    # Assert that they are not empty (assuming sklearn is successfully loaded and indexed)
    assert len(updates["_obj_dict"]) > 0
    assert len(updates["_type_of_objs"]) > 0
    assert len(updates["_objs_by_type"]) > 0
    
    # Assert specific known keys exist to validate the crawl
    assert "RandomForestClassifier" in updates["_obj_dict"]
    assert updates["_obj_dict"]["RandomForestClassifier"] == "sklearn.ensemble.RandomForestClassifier"
    assert updates["_type_of_objs"]["RandomForestClassifier"] == "classifier"
    assert "RandomForestClassifier" in updates["_objs_by_type"]["classifier"]

def test_sklearn_updater_target_path():
    updater = SklearnUpdater()
    target_path = updater.get_target_path()
    assert target_path.endswith("scikit_learn_data.json")
    
    # ensure it's an absolute path
    assert os.path.isabs(target_path)
