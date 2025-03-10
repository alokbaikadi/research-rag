# Research RAG
A simple RAG designed to ask questions of a set of research papers.

## Dependencies
This project uses [Ollama](https://ollama.com) for local LLM deployment. Model selection can be tuned using a .env file.

This project uses [Poetry](https://python-poetry.org) for dependency management. To install prerequisites, install poetry and then use `poetry install` to install dependencies.

## Usage

Environment variables are used to alter the behavior of the program.
| Variable | Description |
| -- | -- |
| LLM_MODEL | Model used for chat interaction |
| EMBEDDING_MODEL | Model used for embeddings |
| PERSIST_DIRECTORY | Directory in whwich to store the chroma_db persistant data |
| DATA_DIR | Directory containing the data upon which to build the RAG |

These can be set via a .env file for convenience.

The `LLM_MODEL` and `EMBEDDING_MODEL` must be pulled and ready within ollama.

The data processing can be run via `python data.py`
A single question chat interaction can be run via `python client.py`


## Docker
The provided docker file in `docker/Dockerfile` can be used to build an image which contains the dependencies for the project.

To build the file, use `docker build -f docker/Dockerfile .` from the project directory.

The docker image uses 2 volumes:
| Volume | Description |
| --     | --          |
| data   | Directory containing the research library |
| models | Persistent directory for downloading the llm models. |

## License
[MIT](https://choosealicense.com/licenses/mit/)
