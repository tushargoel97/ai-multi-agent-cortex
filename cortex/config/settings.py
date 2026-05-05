from enum import StrEnum

from pydantic import BaseModel

from cortex.config.azure_openai import AzureOpenAIConfig
from cortex.config.openai import OpenAIConfig


class Provider(StrEnum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"


class Settings(BaseModel):
    # Identity
    assistant_name: str = "Cortex"

    # Database
    database_url: str = (
        "postgresql+psycopg2://cortex:cortex@localhost:5432/cortex"
    )

    # Embeddings
    embedding_model: str = "text-embedding-3-small"

    # LLM provider
    provider: Provider = Provider.OPENAI
    openai: OpenAIConfig = OpenAIConfig()
    azure_openai: AzureOpenAIConfig = AzureOpenAIConfig()
