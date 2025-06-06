# Git Setup Guide for ClassForge

## Initial Setup

1. **Install Git**
   - Download Git from [git-scm.com](https://git-scm.com/downloads)
   - Follow the installation instructions for your operating system

2. **Configure Git**
   ```bash
   # Set your name and email
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"

   # Set default branch name
   git config --global init.defaultBranch main

   # Configure line endings (for Windows)
   git config --global core.autocrlf true
   ```

## Project Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MINH01-HUB/Classforge_.git
   cd Classforge_
   ```

2. **Set Up Branch Protection**
   - Go to GitHub repository settings
   - Navigate to Branches > Branch protection rules
   - Add rule for `main` branch:
     - Require pull request reviews
     - Require status checks to pass
     - Include administrators

## Development Workflow

1. **Create a New Branch**
   ```bash
   # Create and switch to a new branch
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Make your code changes
   - Test your changes thoroughly

3. **Commit Changes**
   ```bash
   # Stage changes
   git add .

   # Commit with a descriptive message
   git commit -m "feat: add new feature"
   ```

4. **Push Changes**
   ```bash
   # Push to remote
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Go to GitHub repository
   - Click "Compare & pull request"
   - Fill in the PR description
   - Request reviews from team members

## Best Practices

1. **Branch Naming**
   - `feature/` - for new features
   - `bugfix/` - for bug fixes
   - `hotfix/` - for urgent fixes
   - `docs/` - for documentation
   - `refactor/` - for code refactoring

2. **Commit Messages**
   - Use present tense
   - Start with a verb
   - Keep it concise but descriptive
   - Use conventional commits format:
     - `feat:` for new features
     - `fix:` for bug fixes
     - `docs:` for documentation
     - `style:` for formatting
     - `refactor:` for code changes
     - `test:` for tests
     - `chore:` for maintenance

3. **Code Review**
   - Review code before merging
   - Ensure tests pass
   - Check for code style consistency
   - Verify documentation is updated

## Troubleshooting

1. **Common Issues**
   ```bash
   # If you have conflicts
   git pull --rebase origin main

   # If you need to reset to a previous state
   git reset --hard HEAD~1

   # If you need to clean untracked files
   git clean -fd
   ```

2. **Repository Maintenance**
   ```bash
   # Check repository status
   git status

   # Check repository health
   git fsck

   # Clean up repository
   git gc
   ```

## Team Collaboration

1. **Code Review Process**
   - Create pull request
   - Request reviews from team members
   - Address review comments
   - Get approval before merging

2. **Communication**
   - Use PR descriptions for context
   - Comment on specific lines for feedback
   - Keep discussions in GitHub issues/PRs

## Security

1. **Access Control**
   - Use SSH keys for authentication
   - Enable 2FA on GitHub
   - Review team access regularly

2. **Sensitive Data**
   - Never commit sensitive data
   - Use environment variables
   - Add sensitive files to .gitignore

## Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com)
- [Conventional Commits](https://www.conventionalcommits.org)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/) 