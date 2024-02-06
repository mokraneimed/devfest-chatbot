from creds import OPENAI_API_KEY
from config import model, history_interactions, chunks_retrieved, similarity_threshold

from prompt import standalone_prompt, question_prompt
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferWindowMemory


def create_conversational_chain(vectorstore: Pinecone) -> ConversationalRetrievalChain:

    memory = ConversationBufferWindowMemory(
        k=history_interactions,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": chunks_retrieved, "score_threshold": similarity_threshold})
    
    question_llm = ChatOpenAI(
        cache=False,
        temperature=0.0,
        model_name=model,
        openai_api_key=OPENAI_API_KEY)

    doc_chain_llm = ChatOpenAI(
        cache=False,
        temperature=0.0,
        model_name=model,
        openai_api_key=OPENAI_API_KEY)

    question_generator = LLMChain(
        llm=question_llm,
        prompt=standalone_prompt)

    doc_chain = load_qa_chain(
        doc_chain_llm,
        chain_type="stuff",
        prompt=question_prompt)
    
    
    return ConversationalRetrievalChain(
        memory=memory,
        retriever=retriever,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        get_chat_history=lambda h : h,
        return_source_documents=True)