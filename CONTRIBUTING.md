# Contributing to Gemini Reproduction Project

## Team Members
- Mustafa Albahrani
- Mohammed Alkhalifa

## Development Workflow

### Getting Started
1. Clone the repository
2. Run `bash scripts/setup_environment.sh`
3. Activate the virtual environment: `source venv/bin/activate`
4. Configure your cluster settings in `configs/cluster_config.yaml`

### Branch Strategy
- `main`: Stable, working code
- `dev`: Development branch for integration
- `feature/*`: Feature branches for specific components
- `experiment/*`: Experimental code and testing

### Making Changes
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test your changes locally
4. Commit with clear messages: `git commit -m "Add: description of change"`
5. Push to GitHub: `git push origin feature/your-feature-name`
6. Create a pull request for review

### Commit Message Format
- `Add: [description]` - New features or files
- `Fix: [description]` - Bug fixes
- `Update: [description]` - Updates to existing code
- `Refactor: [description]` - Code refactoring
- `Docs: [description]` - Documentation changes
- `Test: [description]` - Test-related changes

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Comment complex logic

### Testing
- Write unit tests for new functionality
- Run tests before committing: `pytest tests/`
- Ensure all tests pass before creating PR

### Documentation
- Update README.md if adding new features
- Update architecture.md for design changes
- Document all configuration options
- Add comments for complex algorithms

## Project Structure

```
src/
├── agents/          - Worker and Root agent implementations
├── checkpointing/   - In-memory checkpoint management
├── training/        - Training loops and infrastructure
└── utils/           - Utility functions and helpers
```

## Communication
- Use GitHub Issues for bug reports and feature requests
- Use GitHub Discussions for questions and ideas
- Tag your teammate in relevant PRs and issues

## Milestones and Tasks
See `docs/milestones.md` for detailed timeline and tasks.

## Questions?
Contact:
- Mustafa: mustafa.albahrani@kaust.edu.sa
- Mohammed: mohammed.alkhalifa@kaust.edu.sa

