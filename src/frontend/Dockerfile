FROM node:20-alpine AS app
WORKDIR /app

# 1) Copy manifests and configure robust npm fetch
COPY package*.json ./
RUN npm config set fetch-retries 5 \
 && npm config set fetch-retry-mintimeout 20000 \
 && npm config set fetch-retry-maxtimeout 120000 \
 && npm config set registry https://registry.npmjs.org/

# 2) Install deps (try npmjs; if that fails, fallback to npmmirror)
RUN npm install --no-fund --no-audit || (npm config set registry https://registry.npmmirror.com && npm install --no-fund --no-audit)

# 3) Copy source & build
COPY . .
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

# 4) Serve with Vite preview (dev convenience)
EXPOSE 5173
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--strictPort", "--port", "5173"]