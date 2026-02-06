FROM python:3.9-slim

WORKDIR /app

# This copies EVERYTHING from your repo into the container
COPY . .

# Now we move INTO the cloud_deploy folder where your files actually are
WORKDIR /app/cloud_deploy

# Install from the requirements file inside that folder
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Run uvicorn pointing to the app.py inside cloud_deploy
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
