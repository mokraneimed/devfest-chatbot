from creds import OPENAI_API_KEY
from config import model

from langchain.chat_models.openai import ChatOpenAI

chatbot = ChatOpenAI(model=model, openai_api_key=OPENAI_API_KEY)