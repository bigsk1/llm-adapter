# Universal LLM Adapter

## Overview

The Universal LLM Adapter is a Python-based tool that provides a unified interface for interacting with various Large Language Models (LLMs) from different providers. It simplifies the process of using multiple LLM APIs by abstracting away the differences in their implementations and offering a consistent way to send requests and receive responses.

## Features

- Support for multiple LLM providers:
  - NVIDIA
  - OpenAI
  - Anthropic
  - Google
  - Ollama (local models)
  - Cohere
  - AI21 Labs
  - Hugging Face
  - Aleph Alpha
  - Replicate
  - Azure OpenAI
  - Amazon Bedrock
- Streaming support for real-time responses
- Easy configuration through environment variables
- Flexible model selection
- Unified chat interface for all providers

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/universal-llm-adapter.git
   cd universal-llm-adapter
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Copy the `.env.example` file to `.env`:
   ```
   cp .env.sample .env
   ```

2. Edit the `.env` file and set your desired configuration:
   - Set the `MODEL_NAME` to the specific model you want to use
   - Set the `PROVIDER` to the corresponding LLM provider
   - Add your API keys for the providers you plan to use
   - Adjust other parameters like `TEMPERATURE`, `TOP_P`, and `MAX_TOKENS` as needed

## Usage

To start the chat interface:

```
python chat_interface.py
```

This will initialize the Universal LLM Adapter with the configured provider and model, and start an interactive chat session in your terminal.

## Adding New Providers

To add support for a new LLM provider:

1. Update the `UniversalLLMAdapter` class in `llm_adapter.py`:
   - Add a new condition in the `_initialize_client` method to initialize the client for the new provider
   - Implement the request sending logic in the `send_request` method

2. Add any necessary API keys or configuration options to the `.env` file

3. Update the `requirements.txt` file if the new provider requires additional dependencies

## Contributing

Contributions to the Universal LLM Adapter are welcome! Please feel free to submit pull requests, create issues, or suggest new features.

## License

This project is licensed under the MIT License

## Acknowledgments

- Thanks to all the LLM providers for their APIs and documentation
- Special thanks to the open-source community for the various libraries used in this project

## Disclaimer

This tool is for educational and research purposes. Ensure you comply with the terms of service of each LLM provider when using their APIs.