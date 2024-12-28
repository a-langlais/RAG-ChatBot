# Ce script créé une interface de commande de chatbot qui sera continuellement à l'écoute de l'utilisateur.
# L'utilisateur peut alors entrer des commandes pour interagir avec le chatbot, les demandes sont passées au RAG-chain et retourne une réponse grâce au LLM choisi.

class Chatbot:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def get_response(self, user_input):
        try:
            response = self.qa_chain({"query": user_input})
            return response['result']
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    from rag_chain import RAGChain
    from document_processor import DocumentProcessor
    from embedding_indexer import EmbeddingIndexer

    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()

    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)

    rag_chain = RAGChain(vectorstore)
    qa_chain = rag_chain.create_chain()

    chatbot = Chatbot(qa_chain)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")
