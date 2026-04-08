# Dockerfile — Lambda container image for daily predictor health check.
#
# Lightweight: numpy/pandas/scipy/boto3 only (~120MB).
# No LightGBM, CatBoost, ArcticDB, or vectorbt.
#
# Build (from repo root):
#   docker build --platform linux/amd64 -f lambda_health/Dockerfile -t alpha-engine-health-check .
#
# Run locally (simulates Lambda):
#   docker run -p 9000:8080 alpha-engine-health-check
#   curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
#        -d '{"dry_run": true}'

FROM public.ecr.aws/lambda/python:3.12

# Copy and install Python requirements first for better layer caching.
COPY requirements-health.txt .
RUN pip install --no-cache-dir -r requirements-health.txt

# Copy application code
COPY ssm_secrets.py .
COPY health_status.py .
COPY analysis/__init__.py analysis/
COPY analysis/production_health.py analysis/
COPY analysis/retrain_alert.py analysis/
COPY lambda_health/handler.py handler.py

# Lambda handler entry point
CMD ["handler.handler"]
