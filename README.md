# LLM Workflow Engine (LWE) Chat xAI Provider plugin

x AI Provider plugin for [LLM Workflow Engine](https://github.com/llm-workflow-engine/llm-workflow-engine)

Access to [xAI](https://x.ai) chat models.

## Installation

### Export API key

Grab an xAI API key from [https://console.x.ai](https://console.x.ai)

Export the key into your local environment:

```bash
export XAI_API_KEY=<API_KEY>
```

### From packages

Install the latest version of this software directly from github with pip:

```bash
pip install git+https://github.com/llm-workflow-engine/lwe-plugin-provider-chat-xai
```

### From source (recommended for development)

Install the latest version of this software directly from git:

```bash
git clone https://github.com/llm-workflow-engine/lwe-plugin-provider-chat-xai.git
```

Install the development package:

```bash
cd lwe-plugin-provider-chat-xai
pip install -e .
```

## Configuration

Add the following to `config.yaml` in your profile:

```yaml
plugins:
  enabled:
    - provider_chat_xai
    # Any other plugins you want enabled...
```

## Usage

From a running LWE shell:

```
/provider chat_xai
/model model_name grok-3
```
