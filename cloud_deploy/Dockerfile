FROM python:3.9-slim

WORKDIR /app

# Copy everything from GitHub
COPY . .

# 1. FIND the requirements file (wherever it is) and install it
RUN pip install --no-cache-dir -r $(find . -name requirements.txt | head -n 1)

# 2. FIND the app.py file and its directory
# We will use a script to start the app so we don't have to hardcode the path
EXPOSE 8000

# This command finds app.py, gets its directory, and starts uvicorn from there
CMD ["sh", "-c", "export APP_PATH=$(find . -name app.py | head -n 1) && \
     export APP_DIR=$(dirname $APP_PATH) && \
     cd $APP_DIR && \
     uvicorn app:app --host 0.0.0.0 --port 8000"]
