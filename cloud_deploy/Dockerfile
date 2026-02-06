FROM python:3.9-slim

WORKDIR /app

# Copy requirements from inside the subfolder
COPY cloud_deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the CONTENTS of cloud_deploy into /app
COPY cloud_deploy/ .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
