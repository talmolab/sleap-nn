Run linting with `black` and `ruff`.

Command:

```
black sleap_nn tests && ruff check --fix sleap_nn tests
```

Then manually fix any remaining errors which cannot be automatically fixed by ruff.