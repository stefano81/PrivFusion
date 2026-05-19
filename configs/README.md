# Experiment Configuration Files

This directory contains YAML configuration files for running consolidation experiments.

## Configuration Structure

Each experiment configuration file defines:

- **datasets**: List of datasets to consolidate
- **cluster**: Configuration for feature clustering agent
- **normalize**: Configuration for feature normalization agent
- **transform**: Configuration for transformation code generation agent
- **experiment**: General experiment settings

## LLM Configuration

### Environment Variables

Set the following environment variables based on your LLM backend:

```bash
# For WatsonX
export WATSONX_APIKEY=your_api_key
export WATSONX_PROJECT_ID=your_project_id

# For RITS (if using custom endpoint)
export RITS_API_KEY=your_api_key
export RITS_MODEL_ENDPOINT=https://your-endpoint.com

# For Ollama (runs locally, no API key needed)
# Just ensure Ollama is installed and running
```

### LLM Backend Options

#### 1. Ollama (Recommended for local development)

```yaml
llm: privfusion.agents.llms.OllamaLLM
args:
  model_name: llama3.2  # or deepseek-r1, qwen2.5, etc.
kwargs:
  temperature: 0
  max_tokens: 5000
```

#### 2. WatsonX

```yaml
llm: privfusion.agents.llms.WatsonXLLM
args:
  model_name: ibm/granite-13b-chat-v2
kwargs:
  temperature: 0
  max_tokens: 5000
```

#### 3. RITS Custom Endpoint

```yaml
llm: privfusion.agents.llms.RitsLLM
args:
  model_endpoint: ${RITS_MODEL_ENDPOINT}  # Use environment variable
  model_name: openai/gpt-oss-120b
kwargs:
  temperature: 0
  max_tokens: 5000
```

## Example Experiments

- **experiment_1.yaml**: Basic COVID-19 dataset consolidation
- **experiment_2.yaml**: Extended COVID-19 consolidation with additional metrics
- **experiment_3.yaml**: Multi-source COVID-19 data alignment
- **experiment_4.yaml**: Advanced consolidation with custom prompts
- **experiment_5.yaml**: Experimental configurations with different models
- **experiment_6.yaml**: Performance optimization experiments
- **experiment_7.yaml**: Large-scale consolidation tests

## Creating Custom Experiments

1. Copy an existing experiment file
2. Modify dataset paths and names
3. Adjust system prompts for your use case
4. Configure LLM backend and parameters
5. Set experiment parameters (max_iter, etc.)

## Best Practices

- **Use environment variables** for API keys and endpoints
- **Start with Ollama** for local testing (no API costs)
- **Use lower temperatures** (0-0.3) for deterministic results
- **Adjust max_tokens** based on dataset complexity
- **Test with small datasets** before scaling up
- **Version control** your experiment configurations

## Security Notes

⚠️ **Never commit API keys or credentials to version control**

- Use `.env` files for sensitive configuration
- Reference environment variables in YAML files
- Keep `.env` files in `.gitignore`
- Use `.env.example` as a template for required variables
