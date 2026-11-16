FROM python:3.13-slim

# Set environment variables for Python behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/appuser

# Create a non-root user with a writable home directory.
RUN groupadd -r appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser

WORKDIR /app

# Copy project configuration and source code
COPY pyproject.toml README.md ./
COPY app/ ./app/
COPY scripts/ ./scripts/

# Install the application and its dependencies
RUN pip install .

# Prepare Streamlit configuration directory and set ownership.
RUN mkdir -p /home/appuser/.streamlit \
    && chown -R appuser:appuser /app /home/appuser

# Switch to the non-root user
USER appuser

# Expose the FastAPI port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]