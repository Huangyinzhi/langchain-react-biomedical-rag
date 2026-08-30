from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from utils.config_handler import rag_conf


def get_chat_model():
    return ChatTongyi(model=rag_conf["chat_model_name"])


def get_embed_model():
    return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])

