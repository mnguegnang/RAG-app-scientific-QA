FROM python:3.10-slim

WORKDIR /app

# Install C++ build tools required by FAISS and Git CI to install the RAG project from GitHub if needed
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first
COPY requirements.txt .

# Install the Python packages using standard pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of  RAG code
COPY . .

# Start the FastAPI ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]