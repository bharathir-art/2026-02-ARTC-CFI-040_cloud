# This tells Docker to use Python 3.9, which WORKS with TensorFlow
FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies inside the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

EXPOSE 8000

# This starts the app INSIDE the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]