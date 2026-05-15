"""Tests for the Consolidator class."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from privfusion.consolidater import Consolidator
from privfusion.data_models import DatasetInformation, SemanticInformation, StructuralInformation


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    return MagicMock()


@pytest.fixture
def sample_datasets():
    """Create sample datasets for testing."""
    # Dataset 1
    df1 = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [25, 30, 35],
            "country": ["USA", "UK", "Canada"],
        },
    )

    info1 = DatasetInformation(
        name="dataset1",
        semantic=SemanticInformation(
            description="Patient data",
            column_types=["http://schema.org/identifier", "http://schema.org/age", "http://schema.org/Country"],
            column_semantic=["Patient identifier", "Patient age", "Country of residence"],
            dataset_topics=["healthcare", "demographics"],
        ),
        structure=StructuralInformation(
            names=["patient_id", "age", "country"],
            types=["int64", "int64", "object"],
            constraints=[],
            description="Patient demographic data",
        ),
        relationships=[],
    )

    # Dataset 2
    df2 = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "years": [28, 32, 40],
            "nation": ["USA", "France", "Germany"],
        },
    )

    info2 = DatasetInformation(
        name="dataset2",
        semantic=SemanticInformation(
            description="Person data",
            column_types=["http://schema.org/identifier", "http://schema.org/age", "http://schema.org/Country"],
            column_semantic=["Person identifier", "Person age in years", "Nation"],
            dataset_topics=["demographics"],
        ),
        structure=StructuralInformation(
            names=["id", "years", "nation"],
            types=["int64", "int64", "object"],
            constraints=[],
            description="Person demographic information",
        ),
        relationships=[],
    )

    return {
        "dataset1": {"data": df1, "info": info1},
        "dataset2": {"data": df2, "info": info2},
    }


def test_consolidator_initialization():
    """Test Consolidator initialization."""
    consolidator = Consolidator()
    assert consolidator._number_consolidations == 0
    assert consolidator._unified_schemas == []


def test_consolidate_basic(mock_llm, sample_datasets):
    """Test basic consolidation workflow."""
    # Mock AgentCluster response
    cluster_result = [
        {"dataset": "dataset1", "feature_name": "patient_id", "cluster_id": 1},
        {"dataset": "dataset1", "feature_name": "age", "cluster_id": 2},
        {"dataset": "dataset1", "feature_name": "country", "cluster_id": 3},
        {"dataset": "dataset2", "feature_name": "id", "cluster_id": 1},
        {"dataset": "dataset2", "feature_name": "years", "cluster_id": 2},
        {"dataset": "dataset2", "feature_name": "nation", "cluster_id": 3},
    ]

    # Mock AgentNorm response
    norm_result = {
        "uri": "http://schema.org/identifier",
        "feature_name": "identifier",
        "value_structure": "integer",
        "transformations": [
            {"dataset": "dataset1", "name": "patient_id", "recommendation": "rename to identifier"},
            {"dataset": "dataset2", "name": "id", "recommendation": "rename to identifier"},
        ],
    }

    with (
        patch("privfusion.consolidater.AgentCluster") as MockAgentCluster,
        patch("privfusion.consolidater.AgentNorm") as MockAgentNorm,
    ):
        mock_cluster_instance = MockAgentCluster.return_value
        mock_cluster_instance.cluster.return_value = cluster_result

        mock_norm_instance = MockAgentNorm.return_value
        mock_norm_instance.normalize.return_value = norm_result

        consolidator = Consolidator()
        result = consolidator.consolidate(sample_datasets, mock_llm)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6  # 3 columns from each dataset
        assert consolidator._number_consolidations == 1
        assert len(consolidator._unified_schemas) == 1


def test_consolidate_with_unclustered_features(mock_llm, sample_datasets):
    """Test consolidation with features that don't cluster (cluster_id=9999)."""
    cluster_result = [
        {"dataset": "dataset1", "feature_name": "patient_id", "cluster_id": 1},
        {"dataset": "dataset1", "feature_name": "age", "cluster_id": 9999},  # Unclustered
        {"dataset": "dataset1", "feature_name": "country", "cluster_id": 3},
        {"dataset": "dataset2", "feature_name": "id", "cluster_id": 1},
        {"dataset": "dataset2", "feature_name": "years", "cluster_id": 9999},  # Unclustered
        {"dataset": "dataset2", "feature_name": "nation", "cluster_id": 3},
    ]

    norm_result = {
        "uri": "http://schema.org/identifier",
        "feature_name": "identifier",
        "value_structure": "integer",
        "transformations": [
            {"dataset": "dataset1", "name": "patient_id", "recommendation": "rename to identifier"},
            {"dataset": "dataset2", "name": "id", "recommendation": "rename to identifier"},
        ],
    }

    with (
        patch("privfusion.consolidater.AgentCluster") as MockAgentCluster,
        patch("privfusion.consolidater.AgentNorm") as MockAgentNorm,
    ):
        mock_cluster_instance = MockAgentCluster.return_value
        mock_cluster_instance.cluster.return_value = cluster_result

        mock_norm_instance = MockAgentNorm.return_value
        mock_norm_instance.normalize.return_value = norm_result

        consolidator = Consolidator()
        result = consolidator.consolidate(sample_datasets, mock_llm)

        # Verify unclustered features are not normalized
        unclustered = result[result.cluster_id == 9999]
        assert len(unclustered) == 2
        assert unclustered["norm_uri"].isna().all()


def test_consolidate_multiple_times(mock_llm, sample_datasets):
    """Test running consolidation multiple times."""
    cluster_result = [
        {"dataset": "dataset1", "feature_name": "patient_id", "cluster_id": 1},
        {"dataset": "dataset1", "feature_name": "age", "cluster_id": 2},
        {"dataset": "dataset1", "feature_name": "country", "cluster_id": 3},
        {"dataset": "dataset2", "feature_name": "id", "cluster_id": 1},
        {"dataset": "dataset2", "feature_name": "years", "cluster_id": 2},
        {"dataset": "dataset2", "feature_name": "nation", "cluster_id": 3},
    ]

    norm_result = {
        "uri": "http://schema.org/identifier",
        "feature_name": "identifier",
        "value_structure": "integer",
        "transformations": [
            {"dataset": "dataset1", "name": "patient_id", "recommendation": "rename to identifier"},
        ],
    }

    with (
        patch("privfusion.consolidater.AgentCluster") as MockAgentCluster,
        patch("privfusion.consolidater.AgentNorm") as MockAgentNorm,
    ):
        mock_cluster_instance = MockAgentCluster.return_value
        mock_cluster_instance.cluster.return_value = cluster_result

        mock_norm_instance = MockAgentNorm.return_value
        mock_norm_instance.normalize.return_value = norm_result

        consolidator = Consolidator()

        # Run consolidation twice
        consolidator.consolidate(sample_datasets, mock_llm)
        consolidator.consolidate(sample_datasets, mock_llm)

        assert consolidator._number_consolidations == 2
        assert len(consolidator._unified_schemas) == 2


def test_consolidate_empty_datasets(mock_llm):
    """Test consolidation with empty datasets dictionary."""
    with patch("privfusion.consolidater.AgentCluster") as MockAgentCluster:
        mock_cluster_instance = MockAgentCluster.return_value
        mock_cluster_instance.cluster.return_value = []

        consolidator = Consolidator()

        # Empty datasets will cause KeyError in consolidation logic
        with pytest.raises((KeyError, ValueError, Exception)):
            consolidator.consolidate({}, mock_llm)
