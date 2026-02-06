FROM python:3.9-slim

WORKDIR /app

# Copy EVERYTHING from your GitHub repo into /app
COPY . .

# Install dependencies by pointing directly to the file inside the subfolder
# This ensures it doesn't matter what the "current" directory is
RUN pip install --no-cache-dir -r cloud_deploy/requirements.txt

# Expose the port
EXPOSE 8000

# Run uvicorn by pointing to the app file inside the cloud_deploy folder
# We use 'cloud_deploy.app:app' to tell Python exactly where the module is
CMD ["uvicorn", "cloud_deploy.app:app", "--host", "0.0.0.0", "--port", "8000"]
