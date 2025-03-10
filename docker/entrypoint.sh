#!/bin/bash

source .venv/bin/activate
ollama serve &> ollama.log &
pid=$!

echo "Waiting for Ollama to start..."
until ollama list > /dev/null 2>&1; do
  sleep 2
done

if ollama list | grep -q "$LLM_MODEL"; then
  echo "Model $LLM_MODEL already available."
else
  echo "Retrieving model: $LLM_MODEL"
  ollama pull "$LLM_MODEL"
  echo "Model $LLM_MODEL is ready!"
fi

if ollama list | grep -q "$EMBEDDING_MODEL"; then
  echo "Model $EMBEDDING_MODEL already available."
else
  echo "Retrieving model: $EMBEDDING_MODEL"
  ollama pull "$EMBEDDING_MODEL"
  echo "Model $EMBEDDING_MODEL is ready!"
fi

python data.py # Load the data
python client.py # Start the interaction
