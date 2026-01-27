FROM public.ecr.aws/lambda/python:3.12

# System deps needed by LightGBM (libgomp)
RUN dnf -y install libgomp && dnf clean all

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model + code + defaults
COPY best_model.joblib ${LAMBDA_TASK_ROOT}/best_model.joblib
COPY route_defaults.json ${LAMBDA_TASK_ROOT}/route_defaults.json
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/lambda_function.py
COPY src ${LAMBDA_TASK_ROOT}/src

# Lambda handler
CMD ["lambda_function.lambda_handler"]
