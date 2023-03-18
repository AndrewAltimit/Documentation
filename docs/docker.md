# Containers

Containers provide a consistent environment for applications by packaging software, dependencies, and configurations into a single, portable unit. However, there are some cases where this consistency might be compromised, particularly when dealing with kernel differences.

## Containers vs Virtual Machines

Containers are lightweight, resource-efficient, and portable, making them suitable for modern, scalable applications. Virtual machines provide strong isolation, full OS support, and hardware emulation but can be resource-intensive and slower to start up. The choice between containers and VMs depends on the specific requirements, infrastructure, and goals of your applications.

### Container Pros/Cons

Pros:

- **Lightweight:** Containers share the host OS kernel, making them lightweight compared to VMs.
- **Fast startup:** Containers can start up in seconds, providing faster application deployment and scaling.
- **Resource efficiency:** Containers consume fewer resources, allowing more applications to run on a single host.
- **Portability:** Containers package application code, dependencies, and configurations, enabling consistent deployment across environments.
- **Isolation:** Containers provide process isolation, which can help in running multiple applications without interference.

Cons:

- **Kernel dependency:** Containers share the host's kernel, which can lead to inconsistencies due to kernel differences and may limit cross-platform compatibility.
- **Security:** Containers have a smaller isolation boundary compared to VMs, which could lead to potential security risks if not properly configured.
- **Limited support for certain applications:** Containers are less suited for running applications that require extensive customization of the underlying OS or kernel modifications.

### Virtual Machine Pros/Cons

Pros:

- **Strong isolation:** VMs provide strong isolation between applications, as each VM runs its own OS, which enhances security.
- **Full OS support:** VMs can run multiple instances of various operating systems, including different versions and distributions, providing greater flexibility.
- **Hardware emulation:** VMs can emulate specific hardware configurations, making it possible to run legacy or platform-specific applications.
- **Mature ecosystem:** VMs have been around for a longer time and have a mature ecosystem, with a wide range of management and monitoring tools available.

Cons:

- **Resource-intensive:** VMs run a full OS stack, which consumes more resources than containers, leading to lower host density.
- **Slow startup:** VMs can take minutes to start, which can impact the speed of application deployment and scaling.
- **Less efficient:** VMs require more storage and resources than containers, as each VM includes its own OS and duplicated libraries.
- **Inconsistent deployment:** VMs do not inherently encapsulate application dependencies and configurations, which can lead to inconsistencies across environments.

### Container Consistency

- **Application dependencies:** Containers bundle all required libraries and dependencies, ensuring that the application runs consistently across different environments.
- **Configuration:** Containers encapsulate the application's configuration, making it easy to reproduce and share across teams and environments.
- **Isolation:** Containers provide process isolation, so applications running in separate containers won't interfere with one another.
- **Portability:** Containers can run on any system with container runtime support, regardless of the host's underlying hardware or operating system.

### Container Inconsistency

- **Kernel differences:** Containers share the host's kernel, which means that they are susceptible to inconsistencies stemming from kernel differences. For example, a container running on a host with an older kernel version may not have access to newer kernel features. Additionally, certain system calls or kernel modules may not be available or compatible across different host systems.
- **Host-specific resources:** Containers can access host resources like filesystems, devices, and network interfaces. However, these resources may not be consistent across different host systems, leading to potential inconsistencies in container behavior.
- **Resource limits and constraints:** Containers can be limited in terms of resources, such as CPU, memory, or I/O. These limits may vary between host systems and can impact the consistency of container performance.
- **Platform-specific features:** Some features, such as hardware acceleration, are platform-specific and may not be consistently available across different host systems. As a result, containers relying on these features may experience inconsistent behavior.

While containers provide a high level of consistency for application dependencies, configuration, isolation, and portability, they can be susceptible to inconsistencies due to kernel differences, host-specific resources, resource limits, and platform-specific features. To minimize these inconsistencies, it is essential to understand the requirements of your application and ensure that the host systems are compatible with the desired container environment.

# Docker

Docker is a platform for developing, shipping, and running applications via containerization technology which packages applications and their dependencies into lightweight and portable containers that can run consistently across different environments. Docker provides tools for building and managing containers, including a Dockerfile syntax for defining container images, a command-line interface for managing containers, and a registry for storing and sharing container images.

## Installing Docker

Follow the official Docker documentation to install Docker on your platform:

- [Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Install Docker on Debian](https://docs.docker.com/engine/install/debian/)
- [Install Docker on Fedora](https://docs.docker.com/engine/install/fedora/)
- [Install Docker on CentOS](https://docs.docker.com/engine/install/centos/)
- [Install Docker on Windows](https://docs.docker.com/docker-for-windows/install/)
- [Install Docker on macOS](https://docs.docker.com/docker-for-mac/install/)

## Common Docker CLI Commands

### Images

- List images: `docker images`
- Pull an image: `docker pull <image>:<tag>`
- Remove an image: `docker rmi <image>:<tag>`

### Containers

- List running containers: `docker ps`
- List all containers: `docker ps -a`
- Run a container: `docker run -it --rm --name <container_name> <image>:<tag>`
- Stop a container: `docker stop <container_name>`
- Remove a container: `docker rm <container_name>`

### Container Logs

- View container logs: `docker logs <container_name>`

### Executing Commands Inside Containers

- Execute a command inside a running container: `docker exec -it <container_name> <command>`

### Building Dockerfiles

- Build an image from a Dockerfile: `docker build -t <image>:<tag> .`
- Push an image to a registry: `docker push <image>:<tag>`

### Docker Compose

- Start a multi-container application: `docker-compose up -d`
- Stop a multi-container application: `docker-compose down`

### Docker Volumes

- Create a volume: `docker volume create <volume_name>`
- List volumes: `docker volume ls`
- Remove a volume: `docker volume rm <volume_name>`

## Writing Dockerfiles
A Dockerfile is a script containing instructions to build a Docker image. It automates the process of creating a container by specifying the base image, configuration, application code, and dependencies. This documentation will cover the basics of writing a Dockerfile, its syntax, and using multistage builds.

### Dockerfile Example

A Dockerfile consists of a series of instructions, each starting with an uppercase keyword followed by arguments. The instructions are executed in the order they appear, and each instruction creates a new layer in the Docker image. Comments can be added using the # symbol.

Here's a simple example:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
```

### Dockerfile Instructions

#### FROM

The **FROM** instruction sets the base image for your Dockerfile. It must be the first instruction in the file. You can use an official image from the Docker Hub or a custom image.

```dockerfile
FROM <image>[:<tag>] [AS <name>]
```

#### WORKDIR

The **WORKDIR** instruction sets the working directory for any subsequent **RUN**, **CMD**, **ENTRYPOINT**, **COPY**, and **ADD** instructions. If the directory does not exist, it will be created.

```dockerfile
WORKDIR <path>
```

#### COPY

The **COPY** instruction copies files or directories from the local filesystem to the container's filesystem.

```dockerfile
COPY <src> <dest>
```

#### ADD

The **ADD** instruction is similar to **COPY**, but it can also download remote files and extract compressed files.

```dockerfile
ADD <src> <dest>
```

#### RUN

The **RUN** instruction executes a command during the build process, creating a new layer.

```dockerfile
RUN <command>
```

#### CMD

The **CMD** instruction provides the default command that will be executed when running a container.

```dockerfile
CMD ["executable", "param1", "param2"]
```

#### ENTRYPOINT

The **ENTRYPOINT** instruction allows you to configure a container that will run as an executable.

```dockerfile
ENTRYPOINT ["executable", "param1", "param2"]
```

#### EXPOSE

The **EXPOSE** instruction informs Docker that the container listens on the specified network ports at runtime.

```dockerfile
EXPOSE <port> [<port>/<protocol>...]
```

#### ENV

The **ENV** instruction sets an environment variable.

```dockerfile
ENV <key>=<value> ...
```

#### ARG

The **ARG** instruction defines a variable that can be passed to the build process using the `--build-arg` flag.

```dockerfile
ARG <name>[=<default value>]
```

### Multistage Builds

Multistage builds allow you to optimize the Dockerfile by using multiple FROM instructions, each with a unique name. This is useful when you need to use multiple images or want to reduce the final image size.

Here's an example:

```dockerfile
# Stage 1: Build the application
FROM node:14 AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Create the final image
FROM nginx:1.19-alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

In this example, the first stage uses the node:14 image to build the application, and the second stage uses the nginx:1.19-alpine image to serve the application. The `COPY --from=build` command copies the built application files from the first stage to the final image. This results in a smaller final image without the build dependencies.

## Best Practices

- Use official base images: Official images are maintained and optimized by the creators of the respective software.
- Be specific with base image tags: Specify an exact version or use a specific tag to avoid breaking changes in the future.
- Keep layers to a minimum: Group related commands together and use a single RUN instruction whenever possible.
- Use .dockerignore file: Exclude unnecessary files from the build context to reduce build time and prevent sensitive data from being included in the image.
- Cache dependencies: Copy dependency files separately from the application code to take advantage of Docker's build cache and avoid unnecessary re-installations.
- Use multi-stage builds: Multi-stage builds can help reduce the final image size by only including the necessary files for the runtime environment.
