# Python 3.11 to match the training runtime. Unpickling a scikit-learn artefact
# across widely separated Python versions is not guaranteed to work.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so that the layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application, the serialised pipeline and its metadata.
COPY . .

EXPOSE 7860

# Gunicorn with four workers. Never the Flask development server.
#   -w 4               four worker processes
#   -b 0.0.0.0:7860    bind on all interfaces
#   --timeout 120      allow time for large batch requests
#   app:superkart_sales_api   module:flask_app_object
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "--timeout", "120", "app:superkart_sales_api"]
