"""
Loaders Package

This package contains document loaders for different file formats.
"""

from .base_loader import BaseDocumentLoader, Document, \
    load_document, \
    get_loader_for_file, \
    detect_file_type

__all__ = [
    'BaseDocumentLoader',
    'Document',
    'load_document',
    'get_loader_for_file',
    'detect_file_type'
]
