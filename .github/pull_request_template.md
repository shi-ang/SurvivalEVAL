# Description

_This is a lightweight pull request template to ensure that the workflow does not break. Here, you can write a brief summary of what this pull request does._

## Updates

1. Update 1
2. Update 2

## Fixes

List the issue fixes that are covered in this pull request.

1. Fix # ...
2. Fix # ...

## Mandatory Checklist

- [ ] I have run all the tests, using `pytest --all`, and this is the log I get:

```
paste the results you get from running pytest tests here.
```

- [ ] All jupyter notebooks are runnable with expected results.

- [ ] I have reformatted and ran `isort .` and `black .` (in this same order) on the codebase.

- [ ] I have updated the README and added or edited any new scripts to the integration tests.

- [ ] I have ensured sufficient coverage on the newly implemented features.

- [ ] I have made sure that both the `requirements.txt` and `pyproject.toml` files are updated according to the newly introduced dependencies.

- [ ] Paste the remaining code TODOs using the command `grep -rI --color=auto --exclude-dir={.git,__pycache__,env_folder,.venv,venv,.cache,output,.github,} 'TODO' .` here, and explain them if necessary:

```
Paste the results you get here
```
