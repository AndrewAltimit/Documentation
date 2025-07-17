---
layout: docs
title: Docker in 5 Minutes
difficulty_level: beginner
section: technology
---

# Docker: Your App's Shipping Container (5 Minute Read)

{% include learning-breadcrumb.html 
   path=site.data.breadcrumbs.technology 
   current="Docker in 5 Minutes"
   alternatives=site.data.alternatives.docker_beginner 
%}

{% include skill-level-navigation.html 
   current_level="beginner"
   topic="Docker"
   intermediate_link="/docs/technology/docker/"
   advanced_link="/docs/technology/kubernetes/"
%}

## What is Docker?

Remember the frustration: "But it works on my computer!" 

**Docker is like a shipping container for software**. Just as shipping containers revolutionized cargo transport (same container works on ships, trains, and trucks), Docker ensures your app runs the same everywhere.

### The Apartment Analogy

Imagine you're moving to a new city:

- **Without Docker**: You ship your furniture piece by piece, hope it fits through the doors, pray the electricity works the same way
- **With Docker**: You ship your entire apartment as-is, with furniture arranged, electricity pre-wired, everything exactly as you had it

Docker packages your app with everything it needs to run, creating a portable "apartment" for your code.

## Why Should You Care?

### The Pizza Delivery Problem

You're opening a pizza restaurant:
- Your pizza recipe works perfectly in YOUR kitchen
- But in another kitchen? Different oven temps, ingredient brands, kitchen layouts...
- Result: Inconsistent pizzas!

**Docker gives every kitchen the EXACT same setup** - same oven, same ingredients, same everything. Perfect pizzas everywhere!

## Containers vs Virtual Machines (The Hotel Analogy)

### Virtual Machines = Entire Hotel Rooms
- Each app gets a full room (operating system, furniture, bathroom)
- Heavy and wasteful (why duplicate everything?)
- Takes minutes to "check in" (boot up)

### Containers = Efficient Capsule Hotels
- Each app gets just what it needs (bed, light, outlet)
- Shares common facilities (operating system kernel)
- "Check in" in seconds

```
Virtual Machine:          Docker Container:
┌─────────────────┐      ┌─────────────────┐
│   Full OS       │      │   Just Your App │
│   All Libraries │      │   + Dependencies│
│   Your App      │      │                 │
├─────────────────┤      ├─────────────────┤
│   Full OS       │      │   Another App   │
│   All Libraries │      │   + Dependencies│
│   Another App   │      │                 │
├─────────────────┤      ├─────────────────┤
│   Hypervisor    │      │   Docker Engine │
├─────────────────┤      ├─────────────────┤
│   Host OS       │      │   Host OS       │
└─────────────────┘      └─────────────────┘
```

## The Magic Recipe: Images and Containers

### Docker Image = Recipe
- Instructions for building your app's environment
- Like a blueprint or cookie cutter
- Shared and downloaded like app store apps

### Docker Container = The Dish
- A running instance created from the image
- Can have many containers from one image
- Like cookies made from the same cutter

## Essential Commands (Your Docker Toolkit)

### Running Apps
```bash
docker run hello-world         # Your first container!
docker run -d -p 80:80 nginx  # Run a web server
```

### Managing Containers
```bash
docker ps                      # What's running?
docker ps -a                   # What ran before?
docker stop [container]        # Stop a container
docker rm [container]          # Delete a container
```

### Working with Images
```bash
docker images                  # List your recipes
docker pull ubuntu            # Download Ubuntu image
docker rmi [image]            # Delete an image
```

## Try This Now! (3 Minutes)

### Exercise 1: Run Your First Container
```bash
# Run a simple container
docker run hello-world

# Run an interactive Ubuntu container
docker run -it ubuntu bash
# Type 'exit' to leave
```

### Exercise 2: Run a Real App
```bash
# Run a web server
docker run -d -p 8080:80 nginx

# Visit http://localhost:8080 in your browser
# You're running a web server with one command!

# See it running
docker ps

# Stop it (use the CONTAINER ID from docker ps)
docker stop [CONTAINER_ID]
```

### Exercise 3: Explore Inside
```bash
# Run an interactive container
docker run -it python:3.9 bash

# Inside the container:
python --version
echo "I'm inside a container!"
exit
```

## The Dockerfile: Your App's Recipe Card

A Dockerfile is like a recipe card for your app:

```dockerfile
# Start with a base (like "preheat oven")
FROM python:3.9

# Set up the kitchen (working directory)
WORKDIR /app

# Gather ingredients (copy files)
COPY . .

# Prep work (install dependencies)
RUN pip install -r requirements.txt

# The main dish (run the app)
CMD ["python", "app.py"]
```

## Common "Aha!" Moments

- **"Containers aren't VMs"** - They're lighter, faster, share the host kernel
- **"Images are blueprints"** - Containers are the actual running instances
- **"Docker != Docker Hub"** - Docker is the tool, Docker Hub is like GitHub for images
- **"Containers are disposable"** - Destroy and recreate them freely

## Real-World Examples

### Development Environment
```bash
# Instead of: "Install Python, PostgreSQL, Redis, configure everything..."
docker-compose up

# Entire dev environment ready in seconds!
```

### Testing Different Versions
```bash
# Test on Python 3.8
docker run -v .:/app python:3.8 python /app/test.py

# Test on Python 3.11
docker run -v .:/app python:3.11 python /app/test.py

# No installation needed!
```

## What NOT to Do

- ❌ Don't store data inside containers (it disappears!)
- ❌ Don't run everything as root in production
- ❌ Don't ignore the image size (nobody wants 5GB containers)
- ❌ Don't put secrets in Dockerfiles (they're visible to everyone)

## Ready for More?

You've containerized your first apps! Ready to dive deeper?

- **[Full Docker Documentation →](docker.html)** - Volumes, networks, Docker Compose
- **[Kubernetes →](kubernetes.html)** - Orchestrating many containers
- **Practice Project**: Dockerize a simple web app you've built

## Quick Reference Card

| Task | Command | Real-World Analogy |
|------|---------|-------------------|
| Run container | `docker run [image]` | Start the microwave |
| List running | `docker ps` | Check what's cooking |
| Stop container | `docker stop [id]` | Turn off the stove |
| Remove container | `docker rm [id]` | Clean the dishes |
| List images | `docker images` | Browse recipe book |
| Build image | `docker build .` | Write new recipe |

---

**Remember**: Docker is just a tool that packages your app with its environment. Start with simple examples, and soon you'll wonder how you ever lived without it. The best way to learn? Docker run something right now!