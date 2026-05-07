# 1. Use a slim Python image
FROM python:3.10-slim

# 2. Set environment variables to ensure Python doesn't buffer logs
ENV PYTHONUNBUFFERED=1

# 3. Set working directory
WORKDIR /app

# 4. Install system dependencies for PDF parsing and building libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 6. --- THE PRO TOUCH: Pre-download the Embedding Model ---
# This saves time on Railway and prevents timeout errors
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 7. Copy the rest of your application code
COPY . .

# 8. FastAPI usually runs on 8000, but Railway gives you a dynamic port
# We use 8080 as a standard
EXPOSE 7860

# 9. Start FastAPI using Uvicorn
# We bind to 0.0.0.0 so it's accessible externally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
