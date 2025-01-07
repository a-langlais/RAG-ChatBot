import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from vectorstore_utility import *

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer le token Hugging Face à partir des variables d'environnement
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Chemin où sera stocké la base de données FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

# Fonction pour définir le prompt personnalisé utilisé par le modèle
def set_custom_prompt(custom_prompt_template):
    """
    Crée un objet PromptTemplate avec le modèle de prompt fourni.
    
    Args:
        custom_prompt_template (str): Le modèle de prompt à utiliser.
    
    Returns:
        PromptTemplate: L'objet PromptTemplate configuré.
    """
    prompt = PromptTemplate(template = custom_prompt_template, input_variables = ["context", "question"])
    return prompt

# Fonction pour charger le modèle de langage (LLM) Hugging Face
def load_llm(huggingface_repo_id, huggingface_token):
    """
    Charge un modèle de langage depuis Hugging Face.
    
    Args:
        huggingface_repo_id (str): L'identifiant du modèle Hugging Face.
        huggingface_token (str): Le token d'authentification Hugging Face.
    
    Returns:
        HuggingFaceEndpoint: L'instance du modèle de langage chargé.
    """
    llm = HuggingFaceEndpoint(
        repo_id = huggingface_repo_id,
        temperature = 0.5,
        model_kwargs = {"token": huggingface_token, "max_length": "512"}
    )
    return llm

def main():
    """
    Fonction principale qui gère l'interaction avec l'utilisateur via Streamlit.
    Elle permet de charger un modèle, récupérer des informations à partir de fichiers PDF,
    et répondre aux questions de l'utilisateur en utilisant la chaîne de récupération (QA).
    """
    # Utilisation du modèle Mistral 7B de Hugging Face
    huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

    st.title("RAG Chatbot personnel")

    # Initialisation de la session de l'utilisateur (pour mémoriser les messages de la conversation)
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    with st.chat_message("assistant"):
        st.write("""Salut, je suis ton assistant personnel ! J'explore les *.pdf que j'ai à disposition pour essayer de répondre au mieux à tes questionnements. 
                    Qu'allons nous découvrir ensemble aujourd'hui ?""")

    # Affichage des messages précédents de la conversation
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Entrée de la question de l'utilisateur
    prompt = st.chat_input("Entrez votre question")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Définition du modèle de prompt personnalisé
        custom_prompt_template = """
            Use the pieces of information provided in the context to answer the user's question in detail. 
            Make sure to elaborate on the key points and provide as much relevant information as possible, 
            drawing from the context you have. Don't use markdown quote, or citation code. Don't start your 
            answer with a unique word like "Answer:".

            If you don't know the answer, just say that you don't know—do not attempt to fabricate an answer. 
            Stick strictly to the provided context.

            Context: {context}
            Question: {question}

            Provide a thorough answer based on the context. Include all important details, but avoid unnecessary 
            elaboration. Start the answer directly without small talk.
        """

        try:
            # Charger le modèle d'embedding pour les documents
            embedding_model = get_embedding_model()
            vectorstore = load_vectorstore(embedding_model)  # Charger ou créer le VectorStore

            if vectorstore is None:
                documents = load_pdf_files("data/")  # Charger les fichiers PDF
                vectorstore = create_and_save_vectorstore(documents, embedding_model)  # Créer et sauvegarder un nouveau VectorStore

            # Créer la chaîne de récupération de type RetrievalQA
            qa_chain = RetrievalQA.from_chain_type(
                llm = load_llm(huggingface_repo_id = huggingface_repo_id, huggingface_token = HUGGINGFACE_TOKEN),
                chain_type = "stuff",
                retriever = vectorstore.as_retriever(search_kwargs = {'k': 3}),
                return_source_documents = True,
                chain_type_kwargs = {'prompt': set_custom_prompt(custom_prompt_template)}
            )

            # Obtenir la réponse à la question de l'utilisateur
            response = qa_chain.invoke({'query': prompt})

            result = response.get("result", "").strip()
            if not result:
                result = "Je suis désolé, je n'ai pas pu trouver de réponse à votre question. Essayez avec une autre question."

            source_documents = response.get("source_documents", [])

            # Extraire les sources et éliminer les doublons
            unique_sources = {
                f"{doc.metadata.get('source', 'Unknown PDF')} - Page {doc.metadata.get('page', 'N/A')}"
                for doc in source_documents
            }

            # Préparer et afficher la réponse avec les sources
            sources_summary = "\n\n".join(sorted(unique_sources))
            result_to_show = f"{result}\n\n**Sources:**\n\n{sources_summary}"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()
