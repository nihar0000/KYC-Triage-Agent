# Alternate deployment path for this project — the primary, zero-cost
# target is Hugging Face Spaces' native Streamlit SDK (see README.md),
# which needs no Dockerfile at all. This one exists so the same codebase
# can also be deployed to any container platform (AWS App Runner, AWS
# ECS Fargate, Google Cloud Run, etc.) without changes, since containers
# are the common denominator across cloud providers.

FROM python:3.12-slim

# Runs as a non-root user inside the container — a basic DevSecOps
# hardening step (a compromised process can't write outside its own
# home directory) that costs nothing and is easy to forget.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /home/appuser/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
