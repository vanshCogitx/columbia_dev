# Dockerfile for Google Product Search (Render deployment)
# -------------------------------------------------
# 1️⃣ Base image – slim Python with apt support
# -------------------------------------------------
FROM python:3.14-slim

# -------------------------------------------------
# 2️⃣ Install system libraries required by Chromium (headless)
# -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatk1.0-data \
    libatspi2.0-0 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxshmfence1 \
    libxss1 \
    libxtst6 \
    lsb-release \
    wget \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# 3️⃣ Create a non‑root user (safer runtime)
# -------------------------------------------------
RUN useradd -m appuser
WORKDIR /app
COPY --chown=appuser:appuser . /app

# -------------------------------------------------
# 4️⃣ Install Python requirements and Playwright browsers
# -------------------------------------------------
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    # Install Chromium and its OS deps (Playwright) at build time
    playwright install chromium && \
    playwright install-deps

# -------------------------------------------------
# 5️⃣ Make Playwright reuse the installed browsers (cached location)
# -------------------------------------------------
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.playwright-browsers
RUN mkdir -p $PLAYWRIGHT_BROWSERS_PATH && \
    cp -r /root/.cache/ms-playwright/* $PLAYWRIGHT_BROWSERS_PATH/

# -------------------------------------------------
# 6️⃣ Increase launch timeout (optional but recommended)
# -------------------------------------------------
ENV PLAYWRIGHT_TIMEOUT_MS=300000   # 5 minutes

# -------------------------------------------------
# 7️⃣ Switch to non‑root user, expose FastAPI port
# -------------------------------------------------
USER appuser
EXPOSE 8000

# -------------------------------------------------
# 8️⃣ Start the FastAPI server (CLI entry point)
# -------------------------------------------------
CMD ["python", "src/main.py", "--serve"]
