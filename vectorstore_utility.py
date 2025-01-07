import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

# Chemin vers le répertoire contenant les fichiers de données
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Fonction pour charger les fichiers PDF depuis un répertoire spécifié
def load_pdf_files(data):
    """
    Charge tous les fichiers PDF d'un répertoire donné.

    Args:
        data (str): Le chemin du répertoire contenant les fichiers PDF.

    Returns:
        list: Une liste de documents extraits des fichiers PDF.
    """
    loader = DirectoryLoader(data, glob='*.pdf')  # Charger tous les fichiers .pdf du répertoire
    documents = loader.load()  # Charger le contenu des fichiers PDF
    return documents

# Fonction pour découper le texte extrait des documents en morceaux (chunks)
def create_chunks(extracted_data):
    """
    Découpe un document en morceaux de texte de taille gérable.

    Args:
        extracted_data (list): Liste de documents extraits des fichiers PDF.

    Returns:
        list: Liste de morceaux de texte (chunks).
    """
    # Initialisation d'un séparateur de texte pour découper en morceaux de taille 500 caractères, avec une chevauchement de 50 caractères
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    return text_splitter.split_documents(extracted_data)  # Découpe les documents en morceaux

# Fonction pour créer un modèle d'embedding Hugging Face
def get_embedding_model():
    """
    Crée un modèle d'embedding en utilisant la bibliothèque Hugging Face.

    Returns:
        HuggingFaceEmbeddings: Un objet modèle Hugging Face configuré pour les embeddings.
    """
    return HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')  # Modèle préentraîné pour les embeddings

# Fonction pour créer et sauvegarder le VectorStore à partir des documents et du modèle d'embedding
def create_and_save_vectorstore(documents, embedding_model):
    """
    Crée un VectorStore à partir des documents et du modèle d'embedding, puis le sauvegarde localement.

    Args:
        documents (list): Liste de documents à transformer en embeddings.
        embedding_model (HuggingFaceEmbeddings): Le modèle d'embedding utilisé pour transformer les documents.

    Returns:
        FAISS: L'objet VectorStore (FAISS) créé et sauvegardé.
    """
    text_chunks = create_chunks(documents)  # Découpe les documents en morceaux
    db = FAISS.from_documents(text_chunks, embedding_model)  # Crée le VectorStore avec FAISS
    db.save_local(DB_FAISS_PATH)  # Sauvegarde le VectorStore localement à l'emplacement spécifié
    return db

# Fonction pour charger un VectorStore existant si disponible
def load_vectorstore(embedding_model):
    """
    Charge un VectorStore existant depuis un fichier local, si disponible.

    Args:
        embedding_model (HuggingFaceEmbeddings): Le modèle d'embedding utilisé pour charger le VectorStore.

    Returns:
        FAISS or None: L'objet VectorStore (FAISS) si trouvé et chargé, sinon None.
    """
    # Vérifier si le répertoire contenant la base de données existe et si le fichier index.faiss est présent
    if os.path.exists(DB_FAISS_PATH) and os.path.exists(os.path.join(DB_FAISS_PATH, "index.faiss")):
        try:
            # Charger le VectorStore depuis le chemin local
            db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization = True)
            return db  # Retourner le VectorStore chargé
        except Exception as e:
            print(f"Erreur lors du chargement du VectorStore : {str(e)}")  # En cas d'erreur, afficher l'exception
            return None # Si erreur, retourner None
    else:
        return None # Si le VectorStore n'existe pas, retourner None
