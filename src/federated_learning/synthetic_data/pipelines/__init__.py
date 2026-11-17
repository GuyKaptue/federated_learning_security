"""Reproducible generation -> validation -> quality -> drift -> metadata pipeline."""
from .generate import generate_and_validate, load_synthetic

__all__ = ["generate_and_validate", "load_synthetic"]
