# Docker Reference Guide

## Overview
Docker is a containerization platform that packages an application and its dependencies into a single, portable unit called a container. Containers share the host OS kernel but run in isolated user spaces, making them lighter than virtual machines while still providing process, filesystem, and network isolation.

## Core Concepts

### Images vs Containers
- An **image** is a read-only template containing application code, runtime, libraries, and dependencies.
- A **container** is a running instance of an image, with its own writable layer on top.
- Images are built in layers (each Dockerfile instruction creates a layer), which enables caching and reuse.

### Dockerfile
A Dockerfile is a text file of instructions used to build an image.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Key instructions:
- `FROM` – base image
- `WORKDIR` – sets the working directory inside the image
- `COPY` / `ADD` – copies files from host into the image
- `RUN` – executes a command at build time (creates a layer)
- `CMD` / `ENTRYPOINT` – defines the default command when the container starts
- `EXPOSE` – documents which port the container listens on (does not publish it)
- `ENV` – sets environment variables
- `ARG` – build-time-only variables

### Multi-stage Builds
Used to keep final images small by separating build-time dependencies from runtime dependencies.

```dockerfile
# Build stage
FROM node:20 AS builder
WORKDIR /app
COPY . .
RUN npm install && npm run build

# Runtime stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
```

## Core Commands

```bash
docker build -t myapp:latest .          # Build an image from Dockerfile
docker run -d -p 8000:8000 myapp:latest # Run container, map host:container port
docker ps                               # List running containers
docker ps -a                             # List all containers (including stopped)
docker logs -f <container>              # Stream container logs
docker exec -it <container> bash        # Shell into a running container
docker stop <container>                 # Gracefully stop
docker rm <container>                   # Remove a stopped container
docker images                           # List local images
docker rmi <image>                      # Remove an image
docker system prune -a                  # Clean up unused images/containers/networks
```

## Docker Compose
Compose defines and runs multi-container applications from a single YAML file.

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - db
    environment:
      DATABASE_URL: postgresql://user:pass@db:5432/appdb
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

```bash
docker compose up -d      # Start all services in background
docker compose down       # Stop and remove containers/networks
docker compose logs -f    # Follow logs across services
docker compose build      # Rebuild images
```

## Networking
- Default `bridge` network provides isolation with NAT to the host.
- User-defined bridge networks give containers DNS resolution by container/service name.
- `host` mode shares the host's network stack directly (no port mapping needed, less isolation).
- `host.docker.internal` lets a container reach services running on the host machine (Docker Desktop).

## Volumes & Persistence
- **Named volumes** are managed by Docker and persist independently of container lifecycle — preferred for databases.
- **Bind mounts** map a host directory directly into the container — useful for local development with live code reload.

```bash
docker run -v mydata:/var/lib/postgresql/data postgres:16   # named volume
docker run -v $(pwd):/app myapp                              # bind mount
```

## Image Size Optimization
- Use slim/alpine base images where possible.
- Combine `RUN` commands to reduce layer count.
- Use multi-stage builds to exclude build tools from the final image.
- Add a `.dockerignore` file to exclude `.git`, `__pycache__`, `node_modules`, virtual environments, and data directories from the build context.

## Common Troubleshooting
- **Container exits immediately**: check `docker logs <container>` for the crash reason; often a missing environment variable or failed entrypoint command.
- **Port already in use**: another process is bound to the host port; find and stop it, or map to a different host port.
- **Build is slow / cache not used**: order Dockerfile instructions so rarely-changing steps (like dependency installation) come before frequently-changing steps (like copying source code).
- **"Cannot connect to the Docker daemon"**: Docker service isn't running, or the current user lacks permission to access the Docker socket.

## Security Notes
- Avoid running containers as root where possible (`USER` instruction in Dockerfile).
- Don't bake secrets (API keys, passwords) into images — use environment variables, secret managers, or Docker secrets instead.
- Regularly scan images for vulnerabilities (e.g. `docker scout` or third-party scanners).
