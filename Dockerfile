# Χρήση lightweight Python image
FROM python:3.11-slim

ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
# Εγκατάσταση απαραίτητων συστημικών βιβλιοθηκών για το FAISS και τα Embeddings
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Ορισμός φακέλου εργασίας
WORKDIR /app

# Αντιγραφή των requirements και εγκατάσταση
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγραφή όλου του κώδικα (ingest.py, main.py και data folder)
COPY . .

# Ορισμός environment variables
ENV PYTHONUNBUFFERED=1

# Από προεπιλογή, το container θα τρέχει τον Agent
CMD ["python", "main.py"]