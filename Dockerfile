FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user app ./app
# train once at build time so the app starts instantly (free hosts sleep and restart often)
RUN python -m app.pretrain
ENV TSAAS_MAX_ROWS=3000
# Render/others inject $PORT; Hugging Face uses 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
