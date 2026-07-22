from __future__ import annotations

from src.guesthouses.data_loader import load_guesthouses_from_backend
from src.guesthouses.document_builder import build_guesthouse_documents
from src.guesthouses.vector_store import synchronize_guesthouse_documents
from src.staff_steps.data_loader import load_staff_recruitments
from src.staff_steps.document_builder import build_staff_recruitment_documents
from src.staff_steps.vector_store import synchronize_staff_recruitment_documents


def reconcile_dynamic_indexes() -> dict[str, dict[str, int]]:
    """Make dynamic Chroma collections match the backend's active DB records."""
    guesthouses = build_guesthouse_documents(load_guesthouses_from_backend())
    staff_recruitments = build_staff_recruitment_documents(load_staff_recruitments())
    return {
        "guesthouse": synchronize_guesthouse_documents(guesthouses),
        "staffStep": synchronize_staff_recruitment_documents(staff_recruitments),
    }
