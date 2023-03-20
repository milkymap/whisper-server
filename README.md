# Whisper Transcription API

Une API asynchrone pour transcrire des fichiers audio en utilisant la technologie Whisper.

## Prérequis

- Python 3.7 ou supérieur
- [ZeroMQ](https://zeromq.org/)
- [FFmpeg](https://www.ffmpeg.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Pydantic](https://pydantic-docs.helpmanual.io/)

## Installation

1. Cloner le repository : `git clone https://github.com/your_username/whisper-transcription.git`
2. Accéder au dossier : `cd whisper-transcription`
3. Installer les dépendances : `pip install -r requirements.txt`
4. Démarrer l'API : `python app.py`

## API Endpoints

### POST /transcription

Transcrire un fichier audio.

#### Corps de la requête

- `dirname` (str) : le nom du dossier contenant le fichier audio.

#### Réponse

- `task_id` (str) : l'identifiant de la tâche de transcription.

#### Codes de statut

- 200 OK : la tâche de transcription a été lancée avec succès.
- 400 Bad Request : le corps de la requête est mal formé ou manquant.
- 500 Internal Server Error : une erreur interne s'est produite.

### GET /monitoring/{task_id}

Obtenir l'état d'avancement d'une tâche de transcription.

#### Paramètres

- `task_id` (str) : l'identifiant de la tâche de transcription.

#### Réponse

- `task_id` (str) : l'identifiant de la tâche de transcription.
- `task_status` (str) : l'état actuel de la tâche de transcription.

#### Codes de statut

- 200 OK : la tâche de transcription existe et son état a été renvoyé avec succès.
- 500 Internal Server Error : une erreur interne s'est produite.

## Fonctionnement interne

L'API utilise la bibliothèque Python [ZeroMQ](https://zeromq.org/) pour communiquer avec une application Whisper externe, qui effectue la transcription du fichier audio. La communication est asynchrone et utilise le protocole de transport TCP.

## License

Ce projet est sous la licence MIT. Voir le fichier LICENSE pour plus de détails.
