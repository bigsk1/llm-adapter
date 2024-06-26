import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai
import ollama
import cohere
from ai21 import AI21Client
from huggingface_hub import InferenceClient
from aleph_alpha_client import Client as AlephAlphaClient
import replicate
import boto3


load_dotenv()

class UniversalLLMAdapter:
    def __init__(self):
        self.model_name = os.getenv('MODEL_NAME')
        self.provider = os.getenv('PROVIDER').lower()
        self.temperature = float(os.getenv('TEMPERATURE', 0.7))
        self.top_p = float(os.getenv('TOP_P', 1.0))
        self.max_tokens = int(os.getenv('MAX_TOKENS', 1024))
        self._initialize_client()

    def _initialize_client(self):
        if self.provider == "nvidia":
            self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv('NVIDIA_API_KEY'))
        elif self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        elif self.provider == "google":
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.client = genai.GenerativeModel(self.model_name)
        elif self.provider == "ollama":
            ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            self.client = ollama.Client(host=ollama_host)
        elif self.provider == "cohere":
            self.client = cohere.Client(os.getenv('COHERE_API_KEY'))
        elif self.provider == "ai21":
            self.client = AI21Client(api_key=os.getenv('AI21_API_KEY'))
        elif self.provider == "huggingface":
            self.client = InferenceClient(token=os.getenv('HUGGINGFACE_API_KEY'))
        elif self.provider == "aleph_alpha":
            self.client = AlephAlphaClient(token=os.getenv('ALEPH_ALPHA_API_KEY'))
        elif self.provider == "replicate":
            self.client = replicate.Client(api_token=os.getenv('REPLICATE_API_KEY'))
        elif self.provider == "azure_openai":
            self.client = openai.AzureOpenAI(
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version="2024-02-15-preview",
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
            )
        elif self.provider == "amazon_bedrock":
            self.client = boto3.client(
                service_name='bedrock-runtime',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name='us-east-1'  # Adjust as needed
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def send_request(self, prompt):
        if self.provider in ["nvidia", "openai", "azure_openai"]:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True
            )
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        elif self.provider == "anthropic":
            with self.client.messages.stream(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            ) as stream:
                for chunk in stream:
                    if chunk.delta.text is not None:
                        yield chunk.delta.text
        elif self.provider == "google":
            response = self.client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_tokens
                ),
                stream=True
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        elif self.provider == "ollama":
            stream = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )
            for chunk in stream:
                yield chunk['message']['content']
        elif self.provider == "cohere":
            response = self.client.chat(
                message=prompt,
                model=self.model_name,
                temperature=self.temperature,
                p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True
            )
            for chunk in response:
                yield chunk.text
        elif self.provider == "ai21":
            response = self.client.completion.create(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True
            )
            for chunk in response:
                yield chunk.data.text
        elif self.provider == "huggingface":
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True
            )
            for chunk in response:
                yield chunk.token.text
        elif self.provider == "aleph_alpha":
            response = self.client.complete(
                prompt=prompt,
                model=self.model_name,
                maximum_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True
            )
            for token in response:
                yield token.completion
        elif self.provider == "replicate":
            for chunk in self.client.stream(
                self.model_name,
                input={"prompt": prompt}
            ):
                yield chunk
        elif self.provider == "amazon_bedrock":
            body = {
                "prompt": prompt,
                "max_tokens_to_sample": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
            response = self.client.invoke_model_with_response_stream(
                modelId=self.model_name,
                body=json.dumps(body)
            )
            for event in response['body']:
                chunk = json.loads(event['chunk']['bytes'].decode())
                yield chunk['completion']