FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY jev_dataops ./jev_dataops
RUN pip install --no-cache-dir . && useradd --create-home --uid 10001 jev && mkdir /data && chown jev:jev /data
USER jev
ENV JEV_DATA_DIR=/data PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["jev-dataops", "serve", "--host", "0.0.0.0", "--data-dir", "/data"]
