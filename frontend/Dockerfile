FROM python:3.10-slim

WORKDIR /app

# Copy only the requirements first
COPY requirements.txt .

# Install Chainlit and HTTPX
RUN pip install --no-cache-dir -r requirements.txt

# Copy Chainlit app.py script
COPY . .

# Start the Chainlit Chat UI server
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8001"]