from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Dans ce script, les embeddings sont crées en utilisant un modèle depuis HuggingFace.
# Les embeddings une fois créés sont stockés dans un vecteur FAISS, qui permet de retrouver du texte similaire rapidement à partir d'une demande.

class EmbeddingIndexer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_vectorstore(self, texts):
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore

if __name__ == "__main__":
    from document_processor import DocumentProcessor

    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()

    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)
    print("Vector store created successfully")
