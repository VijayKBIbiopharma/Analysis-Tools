# The base image is this one line, and is expected to change.
#
# ECP section 2 requires that "the Docker container used comes from the approved
# list of images". That list is not in the onboarding document, so this uses the
# official slim Python image and keeps everything below it portable: no
# distribution-specific packages, no compilers, no apt installs. Swapping this
# FROM for whatever the platform team approves should be the whole change.
FROM python:3.11-slim

# Never write .pyc files or buffer stdout: logs must reach the console promptly,
# because console output is what CloudWatch collects (ECP section A.5).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a code change does not re-resolve the environment.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analysis_tools ./analysis_tools
COPY app.py pyproject.toml ./

# Run as a non-root user. Nothing in the application writes to disk, so the
# filesystem can be read-only; the one exception is Streamlit's own config
# directory, which is given to the user rather than left to default to root's.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /home/appuser/.streamlit \
    && chown -R appuser:appuser /home/appuser /app
USER appuser
ENV HOME=/home/appuser

EXPOSE 8501

# A failing container should be restarted rather than left serving errors.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
