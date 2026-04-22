FROM python:3.10-slim

WORKDIR /app

# Install C++ build tools required by FAISS and Git to clone the RAG source
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first
COPY requirements.txt .

# Install the Python packages using standard pip
RUN pip install --no-cache-dir -r requirements.txt

# Clone the RAG source pinned to a specific commit so that upstream changes
# don't silently break main.py's streaming endpoint between builds.
# To upgrade: verify the new commit's API against main.py, then update this SHA.
ARG RAG_COMMIT=f96c2bd36f7d1d8377ebfc88d6ce9d65322b8b1d
RUN git clone https://github.com/mnguegnang/RAG-for-Scientific-QA /tmp/rag-src \
    && cd /tmp/rag-src && git checkout ${RAG_COMMIT} \
    && mv src /app/src \
    && rm -rf /tmp/rag-src

# Copy the rest of the RAG backend code
COPY . .

# Start the FastAPI ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
