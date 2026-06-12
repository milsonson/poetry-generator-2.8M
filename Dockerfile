FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY generation_forms.py generate.py model.py poem_scorer.py web_app.py ./
COPY static ./static
COPY transformer_poetry.pth vocab.json generation_samples.txt ./

EXPOSE 7860

CMD ["python", "web_app.py"]

