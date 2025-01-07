# PDF RAG Chatbot Personnel

## Description

Ce projet implémente un chatbot personnalisé utilisant un modèle de question-réponse (QA) basé sur la recherche augmentée par des documents (RAG). Le chatbot explore un ou plusieurs fichier(s) PDF pour répondre aux questions de l'utilisateur en s'appuyant sur un VectorStore FAISS pour récupérer les informations pertinentes. Il utilise également un modèle LLM open-source depuis Hugging Face pour le traitement du langage naturel.

## Fonctionnalités

* **Interface Streamlit** : Interface utilisateur simple et interactive pour poser des questions.
* **Chargement de PDF** : Les documents PDF sont chargés depuis un répertoire local pour être utilisés comme contexte pour le chatbot.
* **VectorStore FAISS** : Création et gestion d'un VectorStore pour l'indexation et la recherche rapide dans les documents.
* **Utilisation de modèles Hugging Face** : Chargement d'un modèle Hugging Face pour l'embedding et le traitement des requêtes.
* **Personnalisation des prompts** : Possibilité de définir un prompt personnalisé pour répondre aux questions de manière détaillée.

## Fichiers

```bash
RAG-ChatBot/
│
├── data                    # Dossier local contenant les PDF servant de ressources
├── vectorestore            # Dossier contenant le VectorStore FAISS
├── vectorstore_utility.py  # Ensemble de fonctions pour le traitement des PDF et la création de la BDD
├── rag_chatbot.py          # UI de l'application (streamlit)
├── requirements.txt        # Dépendances Python
├── .env                    # Variable d'environnement (pensez à rajouter votre API Hugging-Face)
├── .gitignore              # Fichier pour ignorer les fichiers inutiles dans Git
└── README.md               # Documentation
```

## Démarrer avec ce projet
### Installation

Dans un premier temps, clonez le dépôt GitHub sur votre machine :

```bash
git clone https://github.com/a-langlais/RAG-ChatBot.git
```

Puis installez les dépendances (préferez la création d'un environnement virtuel avant) :

```bash
pip install -r requirements.txt
```

Veillez à renommer le fichier `.env.example` en `.env` puis l'éditer avec votre token Hugging-Face :

```bash
HUGGINGFACE_TOKEN="votre_token_HF"
```

Pour générer un token Hugging Face, rendez-vous sur Hugging Face, connectez-vous à votre compte puis accèdez à la section Settings > Access Tokens. 
Cliquez sur "New token", choisissez les autorisations souhaitées (READ pour simplement accéder aux modèles), et copiez le token généré pour l'utiliser dans ce projet.

### Utilisation

Mettez tous les fichiers PDF que vous souhaitez explorer dans le dossier `/data`.
Puis, pour lancer le chatbot avec l'application Streamlit :

```python
streamlit run rag_chatbot.py
```

Une fois le premier prompt lancé, l'application alimentera le VectorStore FAISS avec les PDF du dossier `/data`.
Veillez à être clair et détaillé dans les questions pour que le modèle puisse apporter des réponses.
Il est possible d'utiliser des modèles plus performants si vos capacités computationnelles suivent. Les modifications peuvent être directement apportées à `vectorestore_utility.py` et `rag_chatbot.py`.
