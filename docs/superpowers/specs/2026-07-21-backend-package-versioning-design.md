# Backend Package and API Versioning Design

## Scope

Reorganize the active Python backends without changing runtime behavior, database storage, or thread persistence.

## Package layout

```text
cortex/
  api/
    dependencies/
    middleware/
    v1/
      endpoints/
  workflow/

ai/app/
  api/
    dependencies/
    middleware/
    v1/
      endpoints/
```

The Cortex workflow package owns graph composition, routing, node execution, research, specialist review, memory, context, runtime construction, synthesis, and state types. Existing imports through `cortex.workflow` remain valid through the package interface.

Endpoint modules contain only HTTP request and response handling. Runtime state and reusable request dependencies live under `dependencies`; middleware configuration lives under `middleware`. Empty placeholder modules will not be created.

## URL versioning

All application endpoints use `/api/v1` as their public base path. This is a hard cutover with no legacy aliases.

- Cortex LangGraph-compatible endpoints move beneath `/api/v1`.
- Local AI admin, inference, model, and health endpoints move beneath `/api/v1`.
- UI proxies, local-model base URLs, Docker health checks, application configuration, and tests move to the new paths in the same change.

## Compatibility and persistence

Payload schemas and endpoint behavior remain unchanged. Only module locations and URL prefixes change. PostgreSQL is never recreated; service deployment uses targeted image rebuilds and `--no-deps` restarts. Thread counts are checked before and after deployment.

## Verification

- Python compilation and full test suite.
- TypeScript type-check, scoped lint, and production build.
- Docker builds for AI, LangGraph, and UI only.
- Health checks through versioned endpoints.
- Admin API smoke test.
- Two-turn local-model context/search smoke test.
- Persisted thread-count comparison and runtime log scan.
