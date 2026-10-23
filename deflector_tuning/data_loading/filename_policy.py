"""Explicit interpretation policy for metadata encoded in export filenames."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FilenameMetadataPolicy:
    """Whether historical in/out prefixes identify separate measurement ports."""

    preserve_port_side: bool = True


def legacy_filename_policy(dataset_id: str) -> FilenameMetadataPolicy:
    """Preserve legacy full-brazing interpretation at the compatibility boundary.

    New callers can supply an explicit policy through DataLoader instead of
    encoding assembly state in a dataset name.
    """
    return FilenameMetadataPolicy(preserve_port_side="fullbrazing" not in dataset_id.lower())
