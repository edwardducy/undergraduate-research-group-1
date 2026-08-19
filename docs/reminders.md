# Project Reminders

This document lists technical reminders and external service notices that require attention in the future.

## Google Drive Synchronization

- **Public client identifier deprecation** Rclone uses a shared public Google Drive client identifier by default. Google plans to retire shared client identifiers. When Google disables this shared identifier, create a dedicated OAuth client identifier in Google Cloud Console and update the GDRIVE_TOKEN secret in GitHub.
