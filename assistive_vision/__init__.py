"""Dependency-free core of the assistive vision prototype."""

from .announcements import AnnouncementPolicy
from .detections import Detection

__all__ = ["AnnouncementPolicy", "Detection"]
