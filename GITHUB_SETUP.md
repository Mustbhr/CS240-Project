# GitHub Setup Guide

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `CS240-Gemini-Reproduction` (or your preferred name)
3. Description: "Fast Failure Recovery in Distributed Training with In-Memory Checkpoints - Gemini Reproduction for CS240"
4. Set to **Private** (recommended for academic work)
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Link Local Repository to GitHub

After creating the repository, run these commands:

```bash
# Navigate to your project directory
cd /Users/mustafa/Desktop/KAUST/CS240/CS240-Project

# Add GitHub remote (replace with your actual repository URL)
git remote add origin https://github.com/YOUR_USERNAME/CS240-Gemini-Reproduction.git

# Verify the remote was added
git remote -v

# Push to GitHub
git push -u origin main
```

**Example** (replace with your actual username):
```bash
git remote add origin https://github.com/mustafa-albahrani/CS240-Gemini-Reproduction.git
git push -u origin main
```

## Step 3: Add Collaborator (Mohammed)

1. Go to your repository on GitHub
2. Click "Settings" tab
3. Click "Collaborators" in the left sidebar
4. Click "Add people"
5. Enter Mohammed's GitHub username or email: `mohammed.alkhalifa@kaust.edu.sa`
6. Send invitation

## Step 4: Mohammed's Setup

Mohammed should:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CS240-Gemini-Reproduction.git
cd CS240-Gemini-Reproduction

# Set up the environment
bash scripts/setup_environment.sh

# Configure cluster settings
cp configs/cluster_config.yaml.template configs/cluster_config.yaml
# Edit configs/cluster_config.yaml with your details
```

## Working Together

### Branch Strategy
- `main`: Stable, working code
- `dev`: Development integration
- `feature/worker-agent`: Feature branch for worker agent
- `feature/root-agent`: Feature branch for root agent
- etc.

### Daily Workflow

**Before starting work:**
```bash
git pull origin main
git checkout -b feature/your-feature-name
```

**After making changes:**
```bash
git add .
git commit -m "Add: description of changes"
git push origin feature/your-feature-name
```

**Then create a Pull Request on GitHub for review**

### Useful Git Commands

```bash
# Check status
git status

# See changes
git diff

# View commit history
git log --oneline --graph

# Update from main
git checkout main
git pull origin main
git checkout your-branch
git merge main

# Sync with remote
git fetch origin
```

## GitHub Features to Use

### Issues
- Create issues for tasks from milestones
- Label them: `bug`, `enhancement`, `documentation`, `question`
- Assign to team members

### Projects (Optional)
- Create a GitHub Project board
- Track progress with Kanban-style columns
- Link issues to project cards

### Discussions
- Use for questions and ideas
- Share research findings
- Discuss design decisions

## Repository Settings Recommendations

### Branch Protection (Optional but recommended)
1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable:
   - Require pull request reviews before merging
   - Require status checks to pass before merging

### .gitignore
Already configured to ignore:
- Python cache files
- Virtual environments
- Model checkpoints
- Logs and results
- Cluster configuration files (with sensitive info)

## Troubleshooting

### If you get authentication errors:
```bash
# Use SSH instead of HTTPS
git remote set-url origin git@github.com:YOUR_USERNAME/CS240-Gemini-Reproduction.git
```

Or set up a Personal Access Token:
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Give it `repo` scope
4. Use token as password when pushing

### If main branch doesn't exist:
```bash
git branch -M main
git push -u origin main
```

## Next Steps After Setup

1. ✅ Both team members can access the repository
2. ✅ Try making a small change and creating a pull request
3. ✅ Set up your cluster configuration files
4. ✅ Review the milestones in `docs/milestones.md`
5. ✅ Start with Week 1 tasks: Environment setup

## Contact

- Mustafa: mustafa.albahrani@kaust.edu.sa
- Mohammed: mohammed.alkhalifa@kaust.edu.sa

Happy coding! 🚀

