"""
Experiment tracking system using SQLite.
Stores all experiment configurations and results for analysis.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class ExperimentTracker:
    """
    SQLite-based experiment tracking system.

    Schema:
        experiments table:
        - id: Auto-incrementing primary key
        - experiment_name: Unique experiment identifier
        - experiment_type: Type of experiment (rank_ablation, module_ablation, etc.)
        - config_json: Full configuration as JSON
        - metrics_json: All metrics as JSON
        - profiling_json: Profiling data as JSON
        - metadata_json: Additional metadata as JSON
        - status: 'completed', 'failed', 'running'
        - timestamp: When experiment was run
        - duration_seconds: How long it took
    """

    def __init__(self, db_path: str = "results/experiments.db"):
        """
        Initialize experiment tracker.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT UNIQUE NOT NULL,
                experiment_type TEXT NOT NULL,
                config_json TEXT NOT NULL,
                metrics_json TEXT,
                profiling_json TEXT,
                metadata_json TEXT,
                status TEXT DEFAULT 'running',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                duration_seconds REAL
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiment_type
            ON experiments(experiment_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status
            ON experiments(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON experiments(timestamp)
        """)

        conn.commit()
        conn.close()

    def log_experiment(self, result_dict: Dict[str, Any]) -> int:
        """
        Log an experiment to the database.

        Args:
            result_dict: Dictionary from ExperimentResult.to_dict()

        Returns:
            Experiment ID in database
        """
        config = result_dict.get("config", {})
        metrics = result_dict.get("metrics", {})
        profiling = result_dict.get("profiling", {})
        metadata = result_dict.get("metadata", {})

        experiment_name = config.get("experiment_name", "unknown")
        experiment_type = config.get("experiment_type", "unknown")
        status = metadata.get("status", "completed")
        duration = result_dict.get("duration_seconds", None)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO experiments
                (experiment_name, experiment_type, config_json, metrics_json,
                 profiling_json, metadata_json, status, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_name,
                experiment_type,
                json.dumps(config),
                json.dumps(metrics),
                json.dumps(profiling),
                json.dumps(metadata),
                status,
                duration
            ))

            experiment_id = cursor.lastrowid
            conn.commit()
            return experiment_id

        finally:
            conn.close()

    def get_experiment(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific experiment by name.

        Args:
            experiment_name: Unique experiment identifier

        Returns:
            Dictionary with experiment data or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM experiments WHERE experiment_name = ?
            """, (experiment_name,))

            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_dict(cursor, row)

        finally:
            conn.close()

    def get_experiments_by_type(self, experiment_type: str) -> List[Dict[str, Any]]:
        """
        Get all experiments of a specific type.

        Args:
            experiment_type: Type of experiment (e.g., 'rank_ablation')

        Returns:
            List of experiment dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM experiments
                WHERE experiment_type = ?
                ORDER BY timestamp DESC
            """, (experiment_type,))

            rows = cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows]

        finally:
            conn.close()

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """
        Get all experiments.

        Returns:
            List of all experiment dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM experiments ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows]

        finally:
            conn.close()

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics across all experiments.

        Returns:
            Dictionary with aggregate statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Count by type
            cursor.execute("""
                SELECT experiment_type, COUNT(*)
                FROM experiments
                GROUP BY experiment_type
            """)
            type_counts = dict(cursor.fetchall())

            # Count by status
            cursor.execute("""
                SELECT status, COUNT(*)
                FROM experiments
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())

            # Total experiments
            cursor.execute("SELECT COUNT(*) FROM experiments")
            total_count = cursor.fetchone()[0]

            return {
                "total_experiments": total_count,
                "by_type": type_counts,
                "by_status": status_counts,
            }

        finally:
            conn.close()

    def delete_experiment(self, experiment_name: str) -> bool:
        """
        Delete an experiment from the database.

        Args:
            experiment_name: Unique experiment identifier

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                DELETE FROM experiments WHERE experiment_name = ?
            """, (experiment_name,))

            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

        finally:
            conn.close()

    def clear_all_experiments(self):
        """Delete all experiments from the database. Use with caution!"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM experiments")
            conn.commit()

        finally:
            conn.close()

    def export_to_json(self, output_path: str):
        """
        Export all experiments to a JSON file.

        Args:
            output_path: Path to output JSON file
        """
        experiments = self.get_all_experiments()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(experiments, f, indent=2, default=str)

    def _row_to_dict(self, cursor, row) -> Dict[str, Any]:
        """Convert a database row to a dictionary."""
        columns = [desc[0] for desc in cursor.description]
        result = dict(zip(columns, row))

        # Parse JSON fields
        for json_field in ['config_json', 'metrics_json', 'profiling_json', 'metadata_json']:
            if json_field in result and result[json_field]:
                result[json_field.replace('_json', '')] = json.loads(result[json_field])

        return result

    def query_experiments(
        self,
        experiment_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query experiments with filters.

        Args:
            experiment_type: Filter by experiment type
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of matching experiments
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM experiments WHERE 1=1"
            params = []

            if experiment_type:
                query += " AND experiment_type = ?"
                params.append(experiment_type)

            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY timestamp DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows]

        finally:
            conn.close()
