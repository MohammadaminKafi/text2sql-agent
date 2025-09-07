# Text2SQL Frontend

## Dev
1. `npm i`
2. (Optional) export `VITE_API_BASE_URL` if your backend runs elsewhere. By default, same-origin is used. In dev, `/api` is proxied to `http://localhost:8000`.
3. `npm run dev`

## Build
- `npm run build`
- `npm run preview`

## Docker
- Build: `docker build -t text2sql-frontend .`
- Run (with backend reachable at `backend:8000` on a network):
  ```bash
  docker network create appnet || true
  docker run --rm -d --name backend --network appnet -p 8000:8000 your-backend-image
  docker run --rm -d --name text2sql --network appnet -p 8080:80 text2sql-frontend
  # Open http://localhost:8080