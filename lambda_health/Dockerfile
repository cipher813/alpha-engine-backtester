# Dockerfile — Lambda container image for daily predictor health check.
#
# Lightweight: numpy/pandas/scipy/boto3 + alpha-engine-lib (~130MB).
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

# Stage alpha-engine-lib from a local vendor directory.
# infrastructure/deploy_health.sh populates vendor/alpha-engine-lib
# from a sibling checkout before the Docker build. Mirrors the
# alpha-engine-predictor Dockerfile staging pattern — the private repo
# reaches the build context without needing a GitHub PAT or Docker
# build secret. The [flow_doctor] extra pulls flow-doctor; we do not
# need [arcticdb] here (the health Lambda reads S3 only).
COPY vendor/alpha-engine-lib /tmp/alpha-engine-lib

# Copy and install Python requirements first for better layer caching.
COPY requirements-health.txt .
RUN pip install --no-cache-dir /tmp/alpha-engine-lib[flow_doctor] && \
    pip install --no-cache-dir -r requirements-health.txt && \
    rm -rf /root/.cache/pip /tmp/alpha-engine-lib

# Copy application code
COPY ssm_secrets.py .
COPY health_status.py .
COPY preflight.py .
COPY analysis/__init__.py analysis/
COPY analysis/production_health.py analysis/
COPY analysis/retrain_alert.py analysis/
COPY lambda_health/handler.py handler.py

# flow-doctor.yaml ships alongside the handler; handler.py resolves
# the absolute path via LAMBDA_TASK_ROOT and passes it to
# alpha_engine_lib.logging.setup_logging().
COPY flow-doctor.yaml .

# Lambda handler entry point
CMD ["handler.handler"]
