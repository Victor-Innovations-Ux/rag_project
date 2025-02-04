##
## FREE PROJECT, 2025
## RAG_SYSTEM DEMO
## File description:
## Gen query
##

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def answer_query(query: str, vectorstore, openai_api_key: str) -> dict:
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        result = qa_chain.invoke(query)
        return result
    except Exception as e:
        print(f"Error while executing the query: {e}")
        return {"result": "", "source_documents": []}
