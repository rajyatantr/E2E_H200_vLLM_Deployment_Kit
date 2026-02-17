import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage SQLite database for document extraction results."""

    def __init__(self, db_path: str):
        """Initialize database connection."""
        self.db_path = db_path

        # Ensure parent directory exists
        try:
            db_parent = Path(self.db_path).parent
            if not db_parent.exists():
                db_parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {db_parent}")
        except Exception as e:
            logger.error(f"Could not create DB directory: {e}")
            raise

        self.init_database()

    def init_database(self):
        """Create tables if they don't exist and run migrations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_name TEXT NOT NULL,
                file_path TEXT,
                file_size INTEGER,
                conversion_type TEXT NOT NULL,
                content TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                created_date TEXT NOT NULL,
                created_time TEXT NOT NULL
            )
        """)

        # v2 migration: add new columns (idempotent)
        migration_columns = [
            ("pages_succeeded", "INTEGER"),
            ("pages_failed", "INTEGER"),
            ("retries_total", "INTEGER"),
            ("render_time_ms", "INTEGER"),
            ("inference_time_ms", "INTEGER"),
            ("total_time_ms", "INTEGER"),
            ("request_id", "TEXT"),
            ("structured_json", "TEXT"),
            ("schema_used", "TEXT"),
        ]
        for col_name, col_type in migration_columns:
            try:
                cursor.execute(
                    f"ALTER TABLE extraction_results ADD COLUMN {col_name} {col_type}"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def insert_result(self, result: Dict) -> int:
        """Insert extraction result into database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now()
        created_date = now.strftime("%Y-%m-%d")
        created_time = now.strftime("%H:%M:%S")

        metadata = result.get("metadata") or {}

        # Serialize structured_json if present
        structured_json_str = None
        if result.get("structured_json"):
            try:
                structured_json_str = json.dumps(result["structured_json"], ensure_ascii=False)
            except (TypeError, ValueError):
                structured_json_str = None

        cursor.execute("""
            INSERT INTO extraction_results (
                document_name, file_path, file_size, conversion_type,
                content, status, error_message, created_date, created_time,
                pages_succeeded, pages_failed, retries_total,
                render_time_ms, inference_time_ms, total_time_ms,
                request_id, structured_json, schema_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.get("document_name"),
            result.get("file_path"),
            result.get("file_size"),
            result.get("conversion_type", result.get("schema_used", "yaml")),
            result.get("content") or result.get("raw_ocr_content"),
            result.get("status"),
            result.get("error_message"),
            created_date,
            created_time,
            metadata.get("pages_succeeded"),
            metadata.get("pages_failed"),
            metadata.get("retries_total"),
            metadata.get("render_time_ms") or metadata.get("ocr_time_ms"),
            metadata.get("inference_time_ms") or metadata.get("structuring_time_ms"),
            metadata.get("total_time_ms"),
            metadata.get("request_id") or result.get("request_id"),
            structured_json_str,
            result.get("schema_used"),
        ))

        conn.commit()
        result_id = cursor.lastrowid
        conn.close()

        logger.info(f"Inserted result ID {result_id}: {result.get('document_name')}")
        return result_id

    def get_all_results(self, limit: Optional[int] = None) -> List[Dict]:
        """Get all extraction results."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM extraction_results ORDER BY id DESC"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_result_by_id(self, result_id: int) -> Optional[Dict]:
        """Get specific result by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM extraction_results WHERE id = ?", (result_id,))
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def search_results(self, query: str) -> List[Dict]:
        """Search results by document name."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM extraction_results
            WHERE document_name LIKE ?
            ORDER BY id DESC
        """, (f"%{query}%",))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM extraction_results")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM extraction_results WHERE status = 'success'")
        successful = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(file_size) FROM extraction_results")
        total_size = cursor.fetchone()[0] or 0

        cursor.execute("SELECT conversion_type, COUNT(*) FROM extraction_results GROUP BY conversion_type")
        type_counts = dict(cursor.fetchall())

        # v2: average processing times
        cursor.execute("SELECT AVG(total_time_ms) FROM extraction_results WHERE total_time_ms IS NOT NULL")
        avg_time = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM extraction_results WHERE schema_used IS NOT NULL")
        ade_count = cursor.fetchone()[0]

        conn.close()

        return {
            "total_documents": total,
            "successful": successful,
            "failed": total - successful,
            "total_file_size_mb": round(total_size / (1024 * 1024), 2),
            "conversion_type_breakdown": type_counts,
            "avg_processing_time_ms": round(avg_time, 1) if avg_time else None,
            "ade_extractions": ade_count,
        }

    def reset_database(self):
        """Drop and recreate all tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS extraction_results")
        conn.commit()
        conn.close()

        self.init_database()
        logger.info("Database reset completed")
