import sqlite3
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class IngestionTracker:
    """
    Tracks which files have been ingested by storing their SHA-256 hash in a
    local SQLite database.  On subsequent runs, a file is skipped if its
    filename AND hash already exist — meaning the content has not changed.
    If the same filename appears with a different hash, the file is treated
    as modified and will be reprocessed.
    """

    def __init__(self, db_path: str):
        """
        Open (or create) the SQLite database at *db_path* and ensure the
        ``processed_files`` table exists.

        Args:
            db_path: Absolute or relative path to the SQLite database file.
        """
        logger.info(f"IngestionTracker: opening database at {db_path}")
        self.db_path = db_path
        # Add a 20-second timeout to handle volume mount locking
        self.conn = sqlite3.connect(db_path, timeout=20.0)
        self._create_table()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_table(self) -> None:
        """Create the tracking table if it does not already exist."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_files (
                filename       TEXT PRIMARY KEY,
                file_hash      TEXT NOT NULL,
                processed_at   TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(file_path: str) -> str:
        """
        Compute the SHA-256 hex digest of the file at *file_path*.

        Reads the file in 8 KB chunks so large PDFs never need to be
        loaded entirely into memory.

        Args:
            file_path: Path to the file to hash.

        Returns:
            The lowercase hex SHA-256 digest string.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def is_already_processed(self, file_path: str) -> bool:
        """
        Check whether *file_path* has already been processed **and** its
        content has not changed since the last run.

        Args:
            file_path: Path to the file to check.

        Returns:
            ``True`` if the file exists in the database with the same hash
            (unchanged).  ``False`` if the file is new or has been modified.
        """
        current_hash = self.compute_hash(file_path)
        # Extract just the filename for the DB lookup
        filename = Path(file_path).name

        row = self.conn.execute(
            "SELECT file_hash FROM processed_files WHERE filename = ?",
            (filename,),
        ).fetchone()

        if row is None:
            return False  # never seen before

        stored_hash = row[0]
        return stored_hash == current_hash

    def mark_as_processed(self, file_path: str) -> None:
        """
        Record *file_path* and its current SHA-256 hash in the database.

        Uses ``INSERT OR REPLACE`` so both new files and re-ingested
        (hash-changed) files are handled in a single statement.

        Args:
            file_path: Path to the file that was successfully processed.
        """
        filename = Path(file_path).name
        file_hash = self.compute_hash(file_path)
        timestamp = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """
            INSERT OR REPLACE INTO processed_files (filename, file_hash, processed_at)
            VALUES (?, ?, ?)
            """,
            (filename, file_hash, timestamp),
        )
        self.conn.commit()
        logger.info(f"IngestionTracker: recorded {filename} (hash: {file_hash[:12]}...)")

    def remove_from_tracker(self, filename: str) -> bool:
        """
        Delete a file's hash record from the database.

        Args:
            filename: The name of the file to remove (e.g., 'amerigo_2015.pdf').

        Returns:
            True if a record was deleted, False if the filename was not found.
        """
        cursor = self.conn.execute(
            "DELETE FROM processed_files WHERE filename = ?",
            (filename,),
        )
        self.conn.commit()
        deleted_count = cursor.rowcount
        
        if deleted_count > 0:
            logger.info(f"IngestionTracker: removed {filename} from tracking.")
            return True
        else:
            logger.warning(f"IngestionTracker: {filename} not found in database.")
            return False

    def reset(self) -> None:
        """Clear all tracking records. Used when re-ingestion is required (e.g. chunking strategy change)."""
        self.conn.execute("DELETE FROM processed_files")
        self.conn.commit()
        logger.info("IngestionTracker: all tracking records cleared.")

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.conn.close()
        logger.info("IngestionTracker: database connection closed.")
