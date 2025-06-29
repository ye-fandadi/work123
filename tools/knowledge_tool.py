from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

class KnowledgeTool:
    def __init__(self):
        embeddings = OpenAIEmbeddings()
        loader = TextLoader("docs/knowledge_base.txt")
        documents = loader.load()
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        self.qa_chain = RetrievalQA.from_chain_type(llm=None, retriever=self.vectorstore.as_retriever())

    def query(self, question):
        return self.qa_chain.run(question)

