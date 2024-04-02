Using the CLI
===============

The CLI is a simple command line interface that allows you to interact with the toolkit for the most common tasks. It is a good starting point for new users. The CLI supports the following commands:

- `initialize` - Initialize a new workspace
- `test` - Run integration tests for a tracker
- `evaluate` - Run the evaluation stack
- `analyze` - Analyze the results
- `pack` - Package results for submission to the evaluation server (used for competition submissions)

To access the CLI, run `vot` or `python -m vot <command> <args>` in the terminal. The CLI supports the `--help` option to display help for each command.

Workspace initialization
------------------------

To initialize a new workspace, run the following command:

```
vot initialize <workspace>
```

This will create a new workspace in the specified directory. 

Integration tests
-----------------

To run integration tests for a tracker, run the following command:

```
vot test <tracker> <workspace>
```