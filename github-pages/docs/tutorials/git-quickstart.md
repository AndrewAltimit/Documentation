---
layout: tutorial
title: "Git Quick Start"
description: "Learn the essential Git commands to start version controlling your projects"
type: "tutorial"
difficulty: "beginner"
estimated_time: "45 minutes"

# Prerequisites
prerequisites:
  - id: "terminal-basics"
    title: "Command Line Essentials"
    required: true
    description: "Basic familiarity with terminal commands"

# Learning objectives
learning_objectives:
  - "Initialize a Git repository"
  - "Stage and commit changes"
  - "View repository history"
  - "Work with remote repositories"
  - "Understand basic Git workflow"

# Skills gained
skills_gained:
  - "Git basics"
  - "Version control"
  - "Repository management"
  - "Collaboration basics"

# Organization
category: "technology/git"
tags:
  - "git"
  - "version-control"
  - "beginner"

# Features
has_interactive: true
has_exercises: true
toc: true

# Related content
related:
  - id: "branching-basics"
    type: "guide"
    path: "/docs/guides/beginner/branching-basics"
    title: "Introduction to Branching"
    description: "Learn how to work with Git branches"
    relationship: "next"
  
  - id: "git-internals"
    type: "concept"
    path: "/docs/concepts/git-internals"
    title: "How Git Works Under the Hood"
    description: "Deep dive into Git's architecture"
    relationship: "advanced"

# Exercises
exercises:
  - title: "Initialize Your First Repository"
    difficulty: "easy"
    description: |
      Create a new directory called `my-first-repo` and initialize it as a Git repository.
      Add a README.md file with a brief description of your project.
    solution: |
      ```bash
      mkdir my-first-repo
      cd my-first-repo
      git init
      echo "# My First Repository" > README.md
      git add README.md
      git commit -m "Initial commit: Add README"
      ```
  
  - title: "Track Multiple Files"
    difficulty: "medium"
    description: |
      Create three files: `index.html`, `style.css`, and `script.js`.
      Add some basic content to each, then stage and commit them with appropriate messages.
    solution: |
      ```bash
      echo "<h1>Hello Git</h1>" > index.html
      echo "h1 { color: blue; }" > style.css
      echo "console.log('Hello Git!');" > script.js
      
      git add index.html
      git commit -m "Add HTML structure"
      
      git add style.css
      git commit -m "Add styling"
      
      git add script.js
      git commit -m "Add JavaScript functionality"
      ```
---

## Introduction

Welcome to your Git journey! Git is the world's most popular version control system, and in this tutorial, you'll learn the essential commands to start managing your code like a professional developer.

{% capture why_git %}
### Why Learn Git?

Git helps you:
- **Track Changes**: See exactly what changed, when, and why
- **Collaborate**: Work with others without conflicts
- **Experiment Safely**: Try new ideas without breaking working code
- **Recover from Mistakes**: Undo changes and restore previous versions

By the end of this tutorial, you'll have a solid foundation in Git that will serve you throughout your development career.
{% endcapture %}

{% include expandable-section.html 
   title="Why is Git Important?" 
   content=why_git 
   type="example" 
   icon="lightbulb" 
   expanded=true %}

## Setting Up Git

Before we start, let's make sure Git is installed and configured on your system.

### Check Git Installation

Open your terminal and run:

```bash
git --version
```

You should see something like: `git version 2.34.1`

If Git isn't installed, visit [git-scm.com](https://git-scm.com) for installation instructions.

### Configure Your Identity

Git needs to know who you are for commit messages:

{% capture simple_config %}
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```
{% endcapture %}

{% capture advanced_config %}
```bash
# Set your identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set your default editor (optional)
git config --global core.editor "code --wait"  # For VS Code

# Enable color output
git config --global color.ui auto

# Set default branch name
git config --global init.defaultBranch main

# View all settings
git config --global --list
```
{% endcapture %}

{% include code-tabs.html 
   tabs="Simple:simple_config,Advanced:advanced_config"
   language="bash" %}

## Creating Your First Repository

A Git repository (or "repo") is a folder where Git tracks changes. Let's create one!

### Initialize a Repository

```bash
# Create a new directory
mkdir my-project
cd my-project

# Initialize Git
git init
```

🎉 Congratulations! You've just created your first Git repository!

### Understanding the .git Directory

{% capture git_internals %}
When you run `git init`, Git creates a hidden `.git` directory. This contains:

- **objects/**: Where Git stores all your data
- **refs/**: References to commits (branches, tags)
- **HEAD**: Points to the current branch
- **config**: Repository-specific settings
- **hooks/**: Scripts that run at certain points

**Important**: Never manually edit files in `.git` unless you really know what you're doing!
{% endcapture %}

{% include expandable-section.html 
   title="Deep Dive: What's in .git?" 
   content=git_internals 
   type="advanced" 
   icon="microscope" %}

## Making Your First Commit

A commit is a snapshot of your project at a specific point in time.

### The Three States

In Git, files can be in three states:

1. **Working Directory**: Where you make changes
2. **Staging Area**: Where you prepare changes for commit
3. **Repository**: Where Git permanently stores snapshots

### Create and Track a File

```bash
# Create a file
echo "# My Awesome Project" > README.md

# Check status
git status
```

You'll see that `README.md` is "untracked" - Git sees it but isn't tracking changes yet.

### Stage the File

```bash
# Add file to staging area
git add README.md

# Check status again
git status
```

Now the file is staged and ready to be committed!

### Commit the Changes

```bash
# Create a commit with a message
git commit -m "Initial commit: Add README"
```

{% capture commit_messages %}
### Writing Good Commit Messages

Good commit messages are crucial for understanding project history. Follow these guidelines:

**Format**:
```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semi-colons, etc.
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Example**:
```
feat: Add user authentication

Implement login and registration functionality using JWT tokens.
Includes password hashing with bcrypt and session management.

Closes #123
```

**Tips**:
- Use present tense ("Add feature" not "Added feature")
- Keep subject line under 50 characters
- Explain *why* not just *what* in the body
{% endcapture %}

{% include expandable-section.html 
   title="Best Practices: Commit Messages" 
   content=commit_messages 
   type="theory" 
   icon="book" %}

## Working with Changes

Let's learn how to manage ongoing changes in your project.

### Viewing Changes

```bash
# See what files have changed
git status

# See specific changes in files
git diff

# See changes in staged files
git diff --staged
```

### Staging Multiple Files

{% capture staging_patterns %}
```bash
# Stage specific files
git add file1.txt file2.txt

# Stage all .js files
git add *.js

# Stage all changes in a directory
git add src/

# Stage everything (use with caution!)
git add .

# Interactive staging (advanced)
git add -p
```
{% endcapture %}

{% include code-tabs.html 
   tabs="Common Patterns:staging_patterns"
   language="bash" %}

### Viewing History

```bash
# See commit history
git log

# One line per commit
git log --oneline

# With graph visualization
git log --oneline --graph

# Show last 5 commits
git log -5
```

## Working with Remote Repositories

Remote repositories allow you to collaborate and backup your code.

### Adding a Remote

```bash
# Add a remote repository (usually GitHub, GitLab, etc.)
git remote add origin https://github.com/username/repo.git

# Verify the remote
git remote -v
```

### Pushing Changes

```bash
# Push to remote repository
git push -u origin main

# After first push, just use:
git push
```

### Cloning a Repository

```bash
# Clone an existing repository
git clone https://github.com/username/repo.git

# Clone into a specific directory
git clone https://github.com/username/repo.git my-local-name
```

{% capture ssh_setup %}
### Setting Up SSH Keys

Using SSH keys is more secure and convenient than HTTPS:

1. **Generate SSH Key**:
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   ```

2. **Start SSH Agent**:
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

3. **Copy Public Key**:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

4. **Add to GitHub/GitLab**: 
   - Go to Settings → SSH Keys
   - Paste your public key

5. **Use SSH URL**:
   ```bash
   git remote set-url origin git@github.com:username/repo.git
   ```
{% endcapture %}

{% include expandable-section.html 
   title="Advanced: SSH Authentication" 
   content=ssh_setup 
   type="advanced" 
   icon="key" %}

## Common Git Workflows

### The Basic Workflow

1. **Make changes** in your working directory
2. **Stage changes** with `git add`
3. **Commit changes** with `git commit`
4. **Push to remote** with `git push`
5. **Pull updates** with `git pull`

### Undoing Changes

{% capture undo_changes %}
```bash
# Discard changes in working directory
git checkout -- filename.txt

# Unstage a file
git reset HEAD filename.txt

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Amend last commit
git commit --amend -m "New commit message"
```

**⚠️ Warning**: Be careful with `--hard` as it permanently deletes changes!
{% endcapture %}

{% include code-tabs.html 
   tabs="Undo Operations:undo_changes"
   language="bash" %}

## Summary and Next Steps

Congratulations! You've learned the essential Git commands:

- ✅ Initialize repositories with `git init`
- ✅ Stage changes with `git add`
- ✅ Commit snapshots with `git commit`
- ✅ View history with `git log`
- ✅ Work with remotes using `git push` and `git pull`

### Quick Reference

| Command | Description |
|---------|-------------|
| `git init` | Create a new repository |
| `git status` | Check current state |
| `git add <file>` | Stage changes |
| `git commit -m "message"` | Create a commit |
| `git log` | View history |
| `git push` | Upload to remote |
| `git pull` | Download from remote |
| `git clone <url>` | Copy a repository |

### Practice Makes Perfect

The best way to learn Git is by using it! Try these challenges:

1. Create a simple website project and track its development
2. Collaborate on an open source project
3. Use Git for your personal notes or documentation

Remember: everyone makes mistakes with Git - that's how we learn! The important thing is to keep practicing.