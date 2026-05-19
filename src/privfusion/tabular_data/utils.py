"""
Some utility methods for Tabular Data Generation
"""

import json
import logging
import os
from typing import Any

import litellm
import pandas as pd
import yaml
from jinja2 import StrictUndefined, Template

# Define templates
GENERATE_TEMPLATE = """{introduction}\n{principles}\n{examples}\n{generation}"""


def read_data(
    file_path: str,
    sep: str = ",",
    compression: str = "infer",
) -> pd.DataFrame:
    logging.info(f"Data will be loaded from {file_path}")
    df = pd.read_csv(file_path, sep=sep, compression=compression)
    return df


def extract_json_as_dict(json_file: str | dict[str, Any] | list[Any]) -> Any:
    if isinstance(json_file, dict | list):
        return json_file  # If already a dictionary or list, return as-is
    try:
        return json.loads(json_file)  # Try parsing if it's a string
    except (ValueError, json.JSONDecodeError):
        print("JSON decode error")
        print(json_file)
        return None


def prompt_model_rits(
    model: str,
    prompt: str,
    role: str,
    rits_api_endpoint: str,
) -> str:
    rits_api_endpoint = rits_api_endpoint
    response = litellm.completion(
        api_base=rits_api_endpoint,
        model=model,
        messages=[
            {"role": f"{role}", "content": f"{prompt}\n"},
        ],
        extra_headers={"RITS_API_KEY": os.environ["RITS_API_KEY"]},
        api_key="fake-key",
    )

    return response.choices[0].message["content"]


def from_yaml(yaml_path: str) -> str:
    """
    Read the prompt template

    :param yaml_path: Path of yaml file

    :return: Template prompt
    """
    with open(yaml_path, encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)
    template = GENERATE_TEMPLATE.format(**yaml_config)
    return template


def encode_prompt(prompt: str, render_dict: dict[str, Any]) -> str:
    """
    Encode the prompt template

    :param prompt: Template prompt
    :param render_dict: Dictionary of the placeholder's values

    :return: Template prompt with corresponding values in the placeholders
    """
    return Template(prompt).render(render_dict, undefined=StrictUndefined)
