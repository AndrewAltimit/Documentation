---
layout: docs
title: "Git Command Reference"
---

# Git Command Reference

Git is a distributed version control system designed to handle everything from small to very large projects with speed and efficiency. This reference covers essential Git commands and workflows.

## Repository Initialization

### Creating a New Repository
```bash
git init
```
Initializes a new Git repository in the current directory, creating a `.git` subdirectory containing all repository metadata.

### Cloning an Existing Repository
```bash
git clone <repository-url>
git clone <repository-url> <directory-name>
```
Creates a local copy of a remote repository. The second form allows specifying a custom directory name.

## Configuration

### User Configuration
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### View Configuration
```bash
git config --list
git config user.name
```

## Basic Workflow Commands

### Status and Differences
```bash
git status              # Show working tree status
git diff                # Show unstaged changes
git diff --staged       # Show staged changes
git diff HEAD           # Show all changes since last commit
```

### Staging Changes
```bash
git add <file>          # Stage specific file
git add .               # Stage all changes in current directory
git add -A              # Stage all changes in repository
git add -p              # Interactive staging
```

### Committing Changes
```bash
git commit -m "Commit message"       # Commit with inline message
git commit                           # Opens editor for message
git commit -am "Message"             # Stage and commit tracked files
git commit --amend                   # Modify last commit
```

## History and Inspection

### Viewing History
```bash
git log                              # Show commit history
git log --oneline                    # Compact format
git log --graph                      # ASCII graph of branches
git log --stat                       # Include file changes
git log -p                           # Show patches
git log --author="Name"              # Filter by author
git log --since="2 weeks ago"        # Time-based filtering
```

### Examining Commits
```bash
git show                             # Show last commit
git show <commit-hash>               # Show specific commit
git show <commit-hash>:<file>        # Show file at specific commit
```

## Branching and Merging

### Branch Management
```bash
git branch                           # List local branches
git branch -a                        # List all branches
git branch <branch-name>             # Create new branch
git branch -d <branch-name>          # Delete branch (safe)
git branch -D <branch-name>          # Force delete branch
```

### Switching Branches
```bash
git checkout <branch-name>           # Switch to branch
git checkout -b <branch-name>        # Create and switch to branch
git switch <branch-name>             # Modern alternative to checkout
git switch -c <branch-name>          # Create and switch (modern)
```

### Merging
```bash
git merge <branch-name>              # Merge branch into current
git merge --no-ff <branch-name>      # Force merge commit
git merge --abort                    # Abort conflicted merge
```

## Remote Operations

### Remote Management
```bash
git remote                           # List remotes
git remote -v                        # Show remote URLs
git remote add <name> <url>          # Add new remote
git remote remove <name>             # Remove remote
git remote rename <old> <new>        # Rename remote
```

### Synchronization
```bash
git fetch                            # Download remote changes
git fetch <remote>                   # Fetch from specific remote
git pull                             # Fetch and merge
git pull --rebase                    # Fetch and rebase
git push                             # Upload local changes
git push <remote> <branch>           # Push specific branch
git push -u origin <branch>          # Set upstream and push
```

## Undoing Changes

### Working Directory
```bash
git restore <file>                   # Discard changes (Git 2.23+)
git checkout -- <file>               # Discard changes (legacy)
git clean -fd                        # Remove untracked files
```

### Staging Area
```bash
git restore --staged <file>          # Unstage file (Git 2.23+)
git reset HEAD <file>                # Unstage file (legacy)
```

### Commits
```bash
git reset --soft HEAD~1              # Undo commit, keep changes staged
git reset --mixed HEAD~1             # Undo commit, unstage changes
git reset --hard HEAD~1              # Undo commit, discard changes
git revert <commit>                  # Create new commit undoing changes
```

## Stashing

```bash
git stash                            # Save changes temporarily
git stash push -m "Description"      # Stash with message
git stash list                       # List stashes
git stash apply                      # Apply most recent stash
git stash apply stash@{n}            # Apply specific stash
git stash pop                        # Apply and remove stash
git stash drop stash@{n}             # Remove specific stash
git stash clear                      # Remove all stashes
```

## Tags

```bash
git tag                              # List tags
git tag <tag-name>                   # Create lightweight tag
git tag -a <tag-name> -m "Message"   # Create annotated tag
git tag -d <tag-name>                # Delete local tag
git push origin <tag-name>           # Push tag to remote
git push origin --tags               # Push all tags
```

## Advanced Operations

### Rebasing
```bash
git rebase <branch>                  # Rebase current branch
git rebase -i HEAD~n                 # Interactive rebase last n commits
git rebase --continue                # Continue after resolving conflicts
git rebase --abort                   # Cancel rebase
```

### Cherry-picking
```bash
git cherry-pick <commit>             # Apply specific commit
git cherry-pick --continue           # Continue after resolving conflicts
git cherry-pick --abort              # Cancel cherry-pick
```

### Searching
```bash
git grep <pattern>                   # Search in working directory
git log -S <string>                  # Find commits that add/remove string
git log --grep=<pattern>             # Search commit messages
```

## Performance and Maintenance

### Optimization
```bash
git gc                               # Garbage collection
git prune                            # Remove unreachable objects
git fsck                             # Check repository integrity
```

### Large Files
```bash
git lfs track "*.psd"                # Track file type with Git LFS
git lfs ls-files                     # List LFS tracked files
```

## Common Workflows

### Feature Branch Workflow
```bash
git checkout -b feature/new-feature  # Create feature branch
# Make changes
git add .
git commit -m "Add new feature"
git push -u origin feature/new-feature
# Create pull request
```

### Hotfix Workflow
```bash
git checkout main
git checkout -b hotfix/critical-fix
# Make fix
git add .
git commit -m "Fix critical issue"
git checkout main
git merge hotfix/critical-fix
git push
```

## Best Practices

1. **Commit Messages**: Use clear, descriptive messages in imperative mood
2. **Atomic Commits**: Each commit should represent one logical change
3. **Branch Naming**: Use descriptive names (feature/, bugfix/, hotfix/)
4. **Regular Commits**: Commit frequently but meaningfully
5. **Pull Before Push**: Always sync with remote before pushing
6. **Review Before Commit**: Use `git diff --staged` before committing

## Troubleshooting

### Common Issues

**Detached HEAD State**
```bash
git checkout <branch-name>           # Return to branch
git checkout -b <new-branch>         # Create branch from current state
```

**Merge Conflicts**
```bash
# Edit conflicted files manually
git add <resolved-files>
git commit                           # Complete merge
```

**Wrong Branch Commits**
```bash
git cherry-pick <commit>             # Copy commit to correct branch
git reset --hard HEAD~1              # Remove from wrong branch
```

## Additional Resources

- [Pro Git Book](https://git-scm.com/book)
- [Git Documentation](https://git-scm.com/docs)
- [Git Internals](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain)