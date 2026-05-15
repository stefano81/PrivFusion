"""Tests for Agent classes (AgentCluster and AgentNorm)."""

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from privfusion.agents.agent_cluster import AgentCluster
from privfusion.agents.agent_norm import AgentNorm
from privfusion.data_models import DatasetInformation, SemanticInformation, StructuralInformation


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    return llm


@pytest.fixture
def sample_datasets_for_clustering():
    """Create sample datasets for clustering tests."""
    info1 = DatasetInformation(
        name="dataset1",
        semantic=SemanticInformation(
            description="Patient data",
            column_types=["http://schema.org/identifier", "http://schema.org/age"],
            column_semantic=["Patient identifier", "Patient age"],
            dataset_topics=["healthcare"],
        ),
        structure=StructuralInformation(
            names=["patient_id", "age"],
            types=["int64", "int64"],
            constraints=[],
            description="Patient data",
        ),
        relationships=[],
    )

    info2 = DatasetInformation(
        name="dataset2",
        semantic=SemanticInformation(
            description="Person data",
            column_types=["http://schema.org/identifier", "http://schema.org/age"],
            column_semantic=["Person identifier", "Person age"],
            dataset_topics=["demographics"],
        ),
        structure=StructuralInformation(
            names=["id", "years"],
            types=["int64", "int64"],
            constraints=[],
            description="Person data",
        ),
        relationships=[],
    )

    return {
        "dataset1": {"info": info1, "data": pd.DataFrame({"patient_id": [1, 2], "age": [25, 30]})},
        "dataset2": {"info": info2, "data": pd.DataFrame({"id": [1, 2], "years": [28, 32]})},
    }


class TestAgentCluster:
    """Tests for AgentCluster class."""

    def test_initialization(self, mock_llm):
        """Test AgentCluster initialization."""
        system_prompt = "Test prompt"
        kwargs = {"temperature": 0.7}

        agent = AgentCluster(llm=mock_llm, system_prompt=system_prompt, kwargs=kwargs)

        assert agent._llm == mock_llm
        assert agent._system_prompt == system_prompt
        assert agent._kwargs == kwargs

    def test_cluster_success(self, mock_llm, sample_datasets_for_clustering):
        """Test successful clustering."""
        expected_clusters = [
            {"dataset": "dataset1", "feature_name": "age", "cluster_id": 1},
            {"dataset": "dataset1", "feature_name": "patient_id", "cluster_id": 2},
            {"dataset": "dataset2", "feature_name": "id", "cluster_id": 2},
            {"dataset": "dataset2", "feature_name": "years", "cluster_id": 1},
        ]

        mock_llm.chat.return_value = {
            "response": json.dumps(expected_clusters),
            "usage": {"total_tokens": 100},
        }

        agent = AgentCluster(llm=mock_llm, system_prompt="Test prompt", kwargs={})
        result = agent.cluster(sample_datasets_for_clustering)

        assert result == expected_clusters
        assert mock_llm.chat.called

    def test_cluster_with_hallucinated_features(self, mock_llm, sample_datasets_for_clustering):
        """Test clustering when LLM hallucinates non-existent features."""
        # Include a hallucinated feature "fake_column"
        clusters_with_hallucination = [
            {"dataset": "dataset1", "feature_name": "age", "cluster_id": 1},
            {"dataset": "dataset1", "feature_name": "patient_id", "cluster_id": 2},
            {"dataset": "dataset1", "feature_name": "fake_column", "cluster_id": 3},  # Hallucinated
            {"dataset": "dataset2", "feature_name": "id", "cluster_id": 2},
            {"dataset": "dataset2", "feature_name": "years", "cluster_id": 1},
        ]

        mock_llm.chat.return_value = {
            "response": json.dumps(clusters_with_hallucination),
            "usage": {"total_tokens": 100},
        }

        agent = AgentCluster(llm=mock_llm, system_prompt="Test prompt", kwargs={})

        # The agent removes hallucinated features but then validates counts, causing an error
        with pytest.raises((ValueError, Exception)):  # Will raise ValueError or RetryError
            agent.cluster(sample_datasets_for_clustering)

    def test_cluster_missing_features(self, mock_llm, sample_datasets_for_clustering):
        """Test clustering when LLM misses some features."""
        # Missing "age" from dataset1
        incomplete_clusters = [
            {"dataset": "dataset1", "feature_name": "patient_id", "cluster_id": 2},
            {"dataset": "dataset2", "feature_name": "id", "cluster_id": 2},
            {"dataset": "dataset2", "feature_name": "years", "cluster_id": 1},
        ]

        mock_llm.chat.return_value = {
            "response": json.dumps(incomplete_clusters),
            "usage": {"total_tokens": 100},
        }

        agent = AgentCluster(llm=mock_llm, system_prompt="Test prompt", kwargs={})

        with pytest.raises((ValueError, Exception)):  # Will raise ValueError wrapped in RetryError
            agent.cluster(sample_datasets_for_clustering)

    def test_cluster_invalid_json_response(self, mock_llm, sample_datasets_for_clustering):
        """Test clustering with invalid JSON response."""
        mock_llm.chat.return_value = {
            "response": "This is not valid JSON",
            "usage": {"total_tokens": 100},
        }

        agent = AgentCluster(llm=mock_llm, system_prompt="Test prompt", kwargs={})

        with pytest.raises((ValueError, Exception)):  # Will raise ValueError wrapped in RetryError
            agent.cluster(sample_datasets_for_clustering)

    def test_cluster_wrong_feature_names(self, mock_llm, sample_datasets_for_clustering):
        """Test clustering with incorrectly named features."""
        wrong_name_clusters = [
            {"dataset": "dataset1", "feature_name": "age", "cluster_id": 1},
            {"dataset": "dataset1", "feature_name": "wrong_name", "cluster_id": 2},  # Wrong name
            {"dataset": "dataset2", "feature_name": "id", "cluster_id": 2},
            {"dataset": "dataset2", "feature_name": "years", "cluster_id": 1},
        ]

        mock_llm.chat.return_value = {
            "response": json.dumps(wrong_name_clusters),
            "usage": {"total_tokens": 100},
        }

        agent = AgentCluster(llm=mock_llm, system_prompt="Test prompt", kwargs={})

        with pytest.raises((ValueError, Exception)):  # Will raise ValueError wrapped in RetryError
            agent.cluster(sample_datasets_for_clustering)

    def test_cluster_retry_mechanism(self, mock_llm, sample_datasets_for_clustering):
        """Test that clustering retries on failure."""
        # First call fails, second succeeds
        valid_clusters = [
            {"dataset": "dataset1", "feature_name": "age", "cluster_id": 1},
            {"dataset": "dataset1", "feature_name": "patient_id", "cluster_id": 2},
            {"dataset": "dataset2", "feature_name": "id", "cluster_id": 2},
            {"dataset": "dataset2", "feature_name": "years", "cluster_id": 1},
        ]

        mock_llm.chat.side_effect = [
            {"response": "invalid json", "usage": {"total_tokens": 100}},
            {"response": json.dumps(valid_clusters), "usage": {"total_tokens": 100}},
        ]

        agent = AgentCluster(llm=mock_llm, system_prompt="Test prompt", kwargs={})
        result = agent.cluster(sample_datasets_for_clustering)

        assert result == valid_clusters
        assert mock_llm.chat.call_count == 2


class TestAgentNorm:
    """Tests for AgentNorm class."""

    def test_initialization(self, mock_llm):
        """Test AgentNorm initialization."""
        system_prompt = "Normalize prompt"
        kwargs = {"temperature": 0.5}

        agent = AgentNorm(llm=mock_llm, system_prompt=system_prompt, kwargs=kwargs)

        assert agent._llm == mock_llm
        assert agent._system_prompt == system_prompt
        assert agent._kwargs == kwargs

    def test_normalize_success(self, mock_llm):
        """Test successful normalization."""
        cluster_context = [
            {
                "name": "patient_id",
                "dataset": "dataset1",
                "uri": "http://schema.org/identifier",
                "example_data": [1, 2, 3],
            },
            {
                "name": "id",
                "dataset": "dataset2",
                "uri": "http://schema.org/identifier",
                "example_data": [10, 20, 30],
            },
        ]

        expected_result = {
            "uri": "http://schema.org/identifier",
            "feature_name": "identifier",
            "value_structure": "integer",
            "transformations": [
                {"dataset": "dataset1", "name": "patient_id", "recommendation": "rename to identifier"},
                {"dataset": "dataset2", "name": "id", "recommendation": "rename to identifier"},
            ],
        }

        mock_llm.chat.return_value = {
            "response": json.dumps(expected_result),
            "usage": {"total_tokens": 150},
        }

        agent = AgentNorm(llm=mock_llm, system_prompt="Normalize prompt", kwargs={})
        result = agent.normalize(cluster_context)

        assert result == expected_result
        assert mock_llm.chat.called

    def test_normalize_invalid_json(self, mock_llm):
        """Test normalization with invalid JSON response."""
        cluster_context = [
            {"name": "age", "dataset": "dataset1", "uri": "http://schema.org/age", "example_data": [25, 30]},
        ]

        mock_llm.chat.return_value = {
            "response": "Not valid JSON at all",
            "usage": {"total_tokens": 50},
        }

        agent = AgentNorm(llm=mock_llm, system_prompt="Normalize prompt", kwargs={})

        with pytest.raises((ValueError, Exception)):  # Will raise ValueError wrapped in RetryError
            agent.normalize(cluster_context)

    def test_normalize_empty_context(self, mock_llm):
        """Test normalization with empty cluster context."""
        expected_result = {
            "uri": None,
            "feature_name": None,
            "value_structure": None,
            "transformations": [],
        }

        mock_llm.chat.return_value = {
            "response": json.dumps(expected_result),
            "usage": {"total_tokens": 20},
        }

        agent = AgentNorm(llm=mock_llm, system_prompt="Normalize prompt", kwargs={})
        result = agent.normalize([])

        assert result == expected_result

    def test_normalize_retry_mechanism(self, mock_llm):
        """Test that normalization retries on failure."""
        cluster_context = [
            {"name": "age", "dataset": "dataset1", "uri": "http://schema.org/age", "example_data": [25, 30]},
        ]

        expected_result = {
            "uri": "http://schema.org/age",
            "feature_name": "age",
            "value_structure": "integer",
            "transformations": [
                {"dataset": "dataset1", "name": "age", "recommendation": "no change needed"},
            ],
        }

        # First two calls fail, third succeeds
        mock_llm.chat.side_effect = [
            {"response": "bad json", "usage": {"total_tokens": 50}},
            {"response": "{incomplete", "usage": {"total_tokens": 50}},
            {"response": json.dumps(expected_result), "usage": {"total_tokens": 100}},
        ]

        agent = AgentNorm(llm=mock_llm, system_prompt="Normalize prompt", kwargs={})
        result = agent.normalize(cluster_context)

        assert result == expected_result
        assert mock_llm.chat.call_count == 3

    def test_normalize_with_complex_transformations(self, mock_llm):
        """Test normalization with complex transformation recommendations."""
        cluster_context = [
            {
                "name": "birth_date",
                "dataset": "dataset1",
                "uri": "http://schema.org/birthDate",
                "example_data": ["1990-01-01", "1985-05-15"],
            },
            {
                "name": "dob",
                "dataset": "dataset2",
                "uri": "http://schema.org/birthDate",
                "example_data": ["01/01/1990", "15/05/1985"],
            },
        ]

        expected_result = {
            "uri": "http://schema.org/birthDate",
            "feature_name": "birth_date",
            "value_structure": "date (YYYY-MM-DD)",
            "transformations": [
                {"dataset": "dataset1", "name": "birth_date", "recommendation": "already in correct format"},
                {
                    "dataset": "dataset2",
                    "name": "dob",
                    "recommendation": "convert from DD/MM/YYYY to YYYY-MM-DD and rename to birth_date",
                },
            ],
        }

        mock_llm.chat.return_value = {
            "response": json.dumps(expected_result),
            "usage": {"total_tokens": 200},
        }

        agent = AgentNorm(llm=mock_llm, system_prompt="Normalize prompt", kwargs={})
        result = agent.normalize(cluster_context)

        assert result == expected_result
        assert len(result["transformations"]) == 2
