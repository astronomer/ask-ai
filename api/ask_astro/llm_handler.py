import os
from enum import Enum
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from ask_astro.config import AzureOpenAIParams, CohereConfig
from langchain.retrievers.document_compressors import CohereRerank, EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

from ask_astro.settings import (
    CONVERSATIONAL_RETRIEVAL_LLM_CHAIN_DEPLOYMENT_NAME,
    CONVERSATIONAL_RETRIEVAL_LLM_CHAIN_TEMPERATURE,
    CONVERSATIONAL_RETRIEVAL_LOAD_QA_CHAIN_DEPLOYMENT_NAME,
    CONVERSATIONAL_RETRIEVAL_LOAD_QA_CHAIN_TEMPERATURE,
    MULTI_QUERY_RETRIEVER_DEPLOYMENT_NAME,
    MULTI_QUERY_RETRIEVER_TEMPERATURE,
)


# enum with all possible categories
class LLM_CATEGORY(Enum):
    MULTI_QUERY_RETRIEVER = 1
    CONVERSATIONAL_RETRIEVAL_LLM_CHAIN = 2
    CONVERSATIONAL_RETRIEVAL_LOAD_QA_CHAIN = 3


def azure_available():
    east_key = os.getenv("AZURE_OPENAI_USEAST_PARAMS", "< >")
    east2_key = os.getenv("AZURE_OPENAI_USEAST2_PARAMS", "< >")
    print("east_key", east_key, "east2_key", east2_key)
    return "< >" not in east_key and "< >" not in east2_key


def openai_available():
    api_key = os.getenv("OPENAI_API_KEY", "< >")
    return "< >" not in api_key


def cohere_available():
    api_key = os.getenv("COHERE_API_KEY", "< >")
    return "< >" not in api_key


class BaseLLMSelector:
    def get_llm(self, category):
        raise NotImplementedError("get_llm method must be implemented")


class AzureSelector(BaseLLMSelector):
    def __init__(self):
        super().__init__()
        self.parameters_per_category = {
            LLM_CATEGORY.MULTI_QUERY_RETRIEVER: {
                "deployment_name": MULTI_QUERY_RETRIEVER_DEPLOYMENT_NAME,
                "temperature": MULTI_QUERY_RETRIEVER_TEMPERATURE,
            },
            LLM_CATEGORY.CONVERSATIONAL_RETRIEVAL_LLM_CHAIN: {
                "deployment_name": CONVERSATIONAL_RETRIEVAL_LLM_CHAIN_DEPLOYMENT_NAME,
                "temperature": CONVERSATIONAL_RETRIEVAL_LLM_CHAIN_TEMPERATURE,
            },
            LLM_CATEGORY.CONVERSATIONAL_RETRIEVAL_LOAD_QA_CHAIN: {
                "deployment_name": CONVERSATIONAL_RETRIEVAL_LOAD_QA_CHAIN_DEPLOYMENT_NAME,
                "temperature": CONVERSATIONAL_RETRIEVAL_LOAD_QA_CHAIN_TEMPERATURE,
            },
        }

    def get_llm(self, category, **kwargs):
        if category not in self.parameters_per_category:
            raise ValueError(
                "AzureSelector has no parameters for category: {}".format(category)
            )

        parameters = self.parameters_per_category[category].copy()
        parameters.update(AzureOpenAIParams.us_east2)
        parameters.update(kwargs)

        return AzureChatOpenAI(**parameters)


class OpenAISelector(BaseLLMSelector):
    def __init__(self):
        super().__init__()
        self.parameters_per_category = {
            LLM_CATEGORY.MULTI_QUERY_RETRIEVER: {
                "temperature": MULTI_QUERY_RETRIEVER_TEMPERATURE
            },
            LLM_CATEGORY.CONVERSATIONAL_RETRIEVAL_LLM_CHAIN: {
                "temperature": CONVERSATIONAL_RETRIEVAL_LLM_CHAIN_TEMPERATURE
            },
            LLM_CATEGORY.CONVERSATIONAL_RETRIEVAL_LOAD_QA_CHAIN: {
                "temperature": CONVERSATIONAL_RETRIEVAL_LOAD_QA_CHAIN_TEMPERATURE
            },
        }

    def get_llm(self, category, **kwargs):
        if category not in self.parameters_per_category:
            raise ValueError(
                "OpenAISelector has no parameters for category: {}".format(category)
            )

        parameters = self.parameters_per_category[category].copy()
        parameters.update(kwargs)

        return ChatOpenAI(**parameters)


class LLMSelector:
    def get_llm(self, category, **kwargs):
        if azure_available():
            return AzureSelector().get_llm(category, **kwargs)
        elif openai_available():
            return OpenAISelector().get_llm(category, **kwargs)
        else:
            raise ValueError(
                "You can't use any of the available LLMs with your current configuration. Make sure you add the corresponding API keys to your environment variables."
            )


class CompressorSelector:
    def get_compressor(self):
        if cohere_available():
            return CohereRerank(user_agent="langchain", top_n=CohereConfig.rerank_top_n)
        elif openai_available():
            embeddings = OpenAIEmbeddings()
            return EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        else:
            raise ValueError(
                "You can't use any of the available compressors with your current configuration. Make sure you add the corresponding API keys to your environment variables."
            )
