# Changelog

All notable changes to the Decision Trace Reconstructor are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Release workflow publishing to PyPI via trusted publishing, triggered by tag pushes and by manual dispatch (used once to publish the existing v0.1.2 line).

## [0.1.2] - 2026-06-12

### Added

- Vendored OEP contract example (`examples/oep_code_review_agent/`) pinning the generic-jsonl mapping, JSONL projection, and expected feasibility report from the Operational Evidence Plane integration, with `tests/integration/test_example_oep_code_review.py` replaying the ingest-and-reconstruct path so a breaking change to the mapping parser or feasibility shape fails in this repository's CI rather than only downstream.
- CI job that installs `.[all,dev]` and runs the test suite with every optional adapter importable, so adapter dependency breakage is caught in this repository.
- Dependabot configuration for pip dependencies and GitHub Actions.
- CI concurrency group cancelling superseded pull-request runs.

### Changed

- Consolidated the duplicated adapter timestamp coercion helpers into `reconstructor.adapters._time` (plain, millisecond-heuristic, and lenient float-string variants) and the duplicated hashed content-field summary into `reconstructor.adapters._payloads`. Adapter behavior is unchanged: the pinned worked-example outputs remain byte-identical, and the LangSmith and OTLP variants keep their intentionally different semantics.

### Fixed

- `reconstructor.__version__` now matches the released package version; `decision-trace --version` had continued to report 0.1.0 after the v0.1.1 release.

## [0.1.1] - 2026-05-27

### Fixed

- `parse_simple_yaml` now accepts the empty flow-mapping notation `{}` as an empty dict, mirroring the existing handling for `[]` as an empty list. This allows operators to declare "no follow-up absorption" in Generic JSONL mapping configs as `absorb_followups: {}`. Non-empty flow mappings (for example `{a: 1}`) continue to be rejected, but now with an explicit `ValueError` rather than silently coerced to a string. Regression covered in `tests/unit/adapters/generic_jsonl/test_generic_jsonl_utils.py`.

### Added

- Generic JSONL adapter docs (`docs/adapters/generic-jsonl.md`) gained a "Minimal Mapping (No Follow-Up Absorption)" section that links to the Operational Evidence Plane's pinned `mapping.v0.yaml` as a real-world example of the no-`absorb_followups` shape, alongside the existing `examples/generic_jsonl_basic_agent/` example which demonstrates the paired tool-call / tool-result pattern.

## [0.1.0] - 2026-04-28

Initial public release of the Decision Trace Reconstructor: a reproducible toolkit for reconstructing decision traces from agent, orchestration, telemetry, and public-incident evidence.

### Added

- Six-stage reconstruction pipeline: fragment collection, temporal ordering, chain assembly, decision-boundary detection, decision-event schema mapping, and feasibility reporting.
- CLI entry point `decision-trace` with commands for reconstruction, synthetic evaluation, named-incident evaluation, schema export, and adapter-specific ingest flows.
- Ten executable regime adapters: LangSmith/LangGraph, OpenTelemetry GenAI OTLP, AWS Bedrock AgentCore, OpenAI Agents SDK, Anthropic Messages/Computer Use, MCP, CrewAI, Microsoft Agent Framework/AutoGen, Pydantic AI, and Generic JSONL.
- Worked examples and golden-output integration tests for representative agent stacks, plus the `replit_drop_database` public-incident fixture.
- Synthetic scenario generator for the architecture × stack-coverage evaluation matrix and named-incident fixtures.
- Structured output layer with Pydantic v2 models, JSON Schema export, W3C PROV-O JSON-LD, and optional Parquet output.
- Quarto report template rendering evaluation artifacts from `results/*.json`.
- Scientific release metadata and packaging files: `CITATION.cff`, `codemeta.json`, `ro-crate-metadata.json`, `MANIFEST.in`, Apache-2.0 license, and Python 3.11+ package metadata.
- CI and local quality gates for pytest, Ruff, mypy, package build, and metadata validation.

### Changed

- README, report, and release metadata now describe the public artifact set without citing unpublished companion papers.
- Release packaging now includes documentation, examples, results, tests, scripts, and scientific metadata in the source distribution.

### Fixed

- Replaced Python's randomized string `hash()` with a SHA-256-based deterministic seed for reproducible synthetic evaluations across processes, hosts, and Python versions.
- Temporal ordering now respects `parent_trace_id` causality before timestamp tie-breaking, with deterministic fallback for cycles.
- `make check` now includes Ruff format checking, matching the documented local release gate.
