"""
Longitudinal tracking database for NeuroSense AI.
Persists patient visits, multivariable risk scores, and BHI indices over time
using a local SQLite database to detect cognitive and motor rate-of-decline.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "database" / "patient_history.db"


class LongitudinalDatabase:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    name TEXT,
                    age REAL,
                    gender TEXT,
                    created_at TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    timestamp TEXT,
                    visit_number INTEGER,
                    ad_risk REAL,
                    pd_risk REAL,
                    bhi_score REAL,
                    risk_tier TEXT,
                    biomarkers_json TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                )
            """)
            conn.commit()

    def register_patient(self, patient_id: str, name: str, age: float, gender: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO patients (patient_id, name, age, gender, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (patient_id, name, age, gender, datetime.now().isoformat()))
            conn.commit()

    def add_assessment(
        self,
        patient_id: str,
        ad_risk: float | None,
        pd_risk: float | None,
        bhi_score: float,
        risk_tier: str,
        biomarkers: dict[str, Any]
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Determine visit number
            cursor.execute("SELECT COUNT(*) FROM assessments WHERE patient_id = ?", (patient_id,))
            visit_count = cursor.fetchone()[0] + 1

            cursor.execute("""
                INSERT INTO assessments (
                    patient_id, timestamp, visit_number, ad_risk, pd_risk, bhi_score, risk_tier, biomarkers_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                visit_count,
                ad_risk,
                pd_risk,
                bhi_score,
                risk_tier,
                json.dumps(biomarkers)
            ))
            conn.commit()
            return visit_count

    def get_patient_history(self, patient_id: str) -> pd.DataFrame:
        with self._get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT visit_number, timestamp, ad_risk, pd_risk, bhi_score, risk_tier
                FROM assessments
                WHERE patient_id = ?
                ORDER BY visit_number ASC
            """, conn, params=(patient_id,))
            return df

    def get_all_patients(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT patient_id, name, age, gender FROM patients ORDER BY patient_id")
            rows = cursor.fetchall()
            return [{"patient_id": r[0], "name": r[1], "age": r[2], "gender": r[3]} for r in rows]

    def compute_decline_velocity(self, patient_id: str) -> dict[str, Any]:
        """Calculates delta BHI against baseline and raises alerts if rapid decline occurs."""
        history = self.get_patient_history(patient_id)
        if len(history) < 2:
            return {
                "has_baseline": len(history) == 1,
                "delta_bhi": 0.0,
                "trend": "Initial Baseline Established",
                "alert": False
            }
        
        baseline_bhi = history.iloc[0]["bhi_score"]
        current_bhi = history.iloc[-1]["bhi_score"]
        delta_bhi = round(current_bhi - baseline_bhi, 1)

        if delta_bhi <= -10.0:
            trend = "Rapid Decline Detected"
            alert = True
        elif delta_bhi < -3.0:
            trend = "Moderate Decline"
            alert = False
        elif delta_bhi > 3.0:
            trend = "Health Improvement / Recovery"
            alert = False
        else:
            trend = "Stable Cognitive & Motor Health"
            alert = False

        return {
            "has_baseline": True,
            "baseline_bhi": baseline_bhi,
            "current_bhi": current_bhi,
            "delta_bhi": delta_bhi,
            "trend": trend,
            "alert": alert,
            "total_visits": len(history)
        }
