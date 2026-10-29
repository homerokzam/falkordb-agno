"""falkordb-agno: GraphRAG over FalkorDB as an Agno Knowledge, with an Ontology Lifecycle."""

from .chunking import HeadingChunking
from .config import ExtractionConfig, RetrievalConfig
from .extraction.extractor import ExtractionError
from .graph_db import FalkorGraphDb
from .knowledge import FalkorGraphKnowledge
from .ontology.changes import ChangeSet, ChangeSetStatus
from .ontology.errors import (
    ChangeSetStateError,
    OntologyConflictError,
    OntologyDiscoveryError,
    OntologyError,
    OntologyEvolutionError,
    OntologyLockError,
    OntologyValidationError,
    OntologyVersionConflictError,
)
from .ontology.manager import OntologyManager
from .ontology.models import Attribute, EntityType, Ontology, RelationType
from .store.connection import FalkorConnection
from .tools import OntologyTools

__version__ = "26.10.29"

__all__ = [
    "Attribute",
    "ChangeSet",
    "ChangeSetStateError",
    "ChangeSetStatus",
    "EntityType",
    "ExtractionConfig",
    "ExtractionError",
    "FalkorConnection",
    "FalkorGraphDb",
    "FalkorGraphKnowledge",
    "HeadingChunking",
    "Ontology",
    "OntologyConflictError",
    "OntologyDiscoveryError",
    "OntologyError",
    "OntologyEvolutionError",
    "OntologyLockError",
    "OntologyManager",
    "OntologyTools",
    "OntologyValidationError",
    "OntologyVersionConflictError",
    "RelationType",
    "RetrievalConfig",
]
