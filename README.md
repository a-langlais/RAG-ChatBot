# RAG-ChatBot

Ce dépôt implémente un modèle de génération augmentée par récupération (RAG) pour construire un chatbot interactif personnel. 
Le chatbot utilise des documents comme source de connaissance et récupère des informations pertinentes pour générer des réponses précises grâce à un ou plusieurs LLM(s) via leur clé API.

## Fonctionnalités

* Traitement de documents : Gestion des documents pour le chatbot.
* Indexation par embeddings : Indexation des documents pour une récupération efficace.
* Chaîne RAG : Intégration de la récupération et de la génération pour créer des réponses.
* Interface Streamlit : Application web simple pour interagir avec le chatbot.

## Fichiers

```
RAG-ChatBot/
│
├── chatbot.py            # Logique principale du chatbot
├── document_processor.py # Traitement et chargement des documents
├── embedding_indexer.py  # Indexation des documents avec des embeddings
├── rag_chain.py          # Connexion entre la récupération et la génération
├── requirements.txt      # Dépendances Python
└── streamlit_app.py      # UI Streamlit
```

## Installation

Clonez le dépôt et installez les dépendances :

```bash
git clone https://github.com/a-langlais/RAG-ChatBot.git
cd RAG-ChatBot
pip install -r requirements.txt
```

Veillez à configurer les variables d'environnement nécessaires en modifiant le fichier .env avec votre/vos clés API.

## Utilisation

Pour lancer le chatbot avec l'application Streamlit :

```bash
streamlit run streamlit_app.py
```

## Personnalisation

* Ajoutez vos propres documents à traiter par le chatbot (format *.txt).
* Modifiez les fichiers `embedding_indexer.py` et `rag_chain.py` pour ajuster l'interaction avec les documents et la génération des réponses.

## Licence

Ce projet est sous licence MIT.


