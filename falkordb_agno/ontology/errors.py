"""Ontology lifecycle errors."""

from __future__ import annotations


class OntologyError(Exception):
    """Base class for ontology errors."""


class OntologyValidationError(OntologyError, ValueError):
    """The ontology (or a change to it) is structurally invalid."""


class OntologyConflictError(OntologyError, ValueError):
    """A declared ontology conflicts with the persisted one."""


class OntologyVersionConflictError(OntologyError):
    """A changeset was based on an outdated ontology version."""


class ChangeSetStateError(OntologyError):
    """Invalid changeset state transition."""


class OntologyLockError(OntologyError):
    """The ontology lock is held by someone else."""


class OntologyEvolutionError(OntologyError, RuntimeError):
    """A migration or backfill failed; no new version was created."""

    def __init__(self, message: str, failed_chunks: list[str] | None = None) -> None:
        super().__init__(message)
        self.failed_chunks = failed_chunks or []


class OntologyDiscoveryError(OntologyError, RuntimeError):
    """The LLM could not produce a valid proposal."""

    def __init__(self, chunk_id: str, attempts: int, last_error: str) -> None:
        super().__init__(f"chunk {chunk_id}: {attempts} attempts failed: {last_error}")
        self.chunk_id, self.attempts, self.last_error = chunk_id, attempts, last_error
