# Trainer helpers

Operational scripts live here so the trainer root remains focused on package,
configuration, data, and documentation entry points.

- `setup.sh` vendors the pinned or requested llama.cpp converter source.
- `restart.sh` safely replaces only the process listening on the trainer port.

Use the stable compatibility commands from the repository documentation:

```bash
bash trainer/setup.sh
cd trainer && ./restart.sh
```

The root scripts forward arguments and environment variables to these helpers.
