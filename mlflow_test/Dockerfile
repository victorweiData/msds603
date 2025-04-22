# Use a lightweight Python image
FROM python:3.8-slim

# Install mlflow
RUN pip install mlflow

# Expose the port mlflow will run on
EXPOSE 5001

# Command to run mlflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001"]