"""
associate_roster.py
The admin-curated master list of valid associate names — used to populate
the searchable "pick an associate" widgets while auditing (tracker Lead
Owner, batch link labels) so names stay consistent instead of free-typed
and fragmenting associate analytics ("Sharukh" vs "Shahrukh" vs "sharukh").

Reading the list is open to any logged-in user (they need it while
auditing); adding/removing is admin-only — see api/main.py's
/api/v1/associate-roster routes and the require_admin dependency.
"""
from __future__ import annotations
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "associate_roster.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_local = threading.local()

# Seeded once, on first run only — the roster you gave us to start with.
_SEED_NAMES = [
    "Muthukumar K", "Hemalatha R", "Mythili Prakash", "Mathan Kumar K", "Yuvashree M",
    "Mohamed Riswan J", "Treeshaa A", "Gothavari P", "Sharon Ken Rose K", "S. Ayesha Sumaiya",
    "Ayyappan J", "Cyril S", "Sudharsan N", "Premalatha V", "Tamil Arasan M",
    "Chris Aaron", "Sneha Padmanabhan", "Farthun Farhana", "Praveen GP", "Abdul Kader Shanavas J",
    "Aishwarya M", "Ragavi K", "Siva Sandhiya P", "Vaishnavi S", "Gayathri Devi B",
    "Akshay Mathew", "Aparna T Pillai", "Ajin K Joji", "Keshava", "Mehataj.K",
    "Shivaranjini. C", "Aman Kumar", "RUPALI SINHA", "Almaz Balbatti", "Rashmi Kumari",
    "Shashikant", "Akanksha", "Abhishek Vishwakarma", "Dave Vishvesh", "Apeksha Mishra",
    "Shashank Shubham", "Madiha M Gadiwan", "Soumyadeep Das", "Rohit Anand", "Ruchit Niraj Singh Deo",
    "Shrawik Kumar Singh", "Rahul Kumar Samal", "Suraj Kumar", "Sania Hashmi", "Himanshu Bhati",
    "Sharik Ashad Laskar", "Roopashree Bv", "H L Srilaxmi", "Rakshith R", "Nandita Suresh Goud K",
    "Sindhu N M", "Kruthik R", "Ashwitha", "Zuhaib Mohammedi Ghafoor", "Kavya Talawar",
    "Darshan S", "Seema Dalal", "Aditya Hanchate", "Anusha M M", "Sushmitha K",
    "Ganavi Nagaraj", "Manoj Adavisiddeshwar Masti", "Karthik R", "Nandhinipriya D", "Divyapriya M",
    "Thrisha M", "Keerthana N", "Pavithra M", "Nadhidarshan A M", "Dhineshkumar K",
    "Indrayan Mitra", "Donald Colin", "Abhinav Kumar Jha", "Vinay manikanta reddy challa", "Aman Tiwari",
    "Harshath S", "Bharath Patil L", "Lokesh Sathiyamoorthy", "Sri Priya",
]


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_db() -> None:
    db = _conn()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS associate_roster (
        name       TEXT PRIMARY KEY,
        added_at   TEXT NOT NULL
    );
    """)
    db.commit()
    if not db.execute("SELECT 1 FROM associate_roster LIMIT 1").fetchone():
        now = datetime.utcnow().isoformat()
        db.executemany(
            "INSERT OR IGNORE INTO associate_roster (name, added_at) VALUES (?,?)",
            [(n, now) for n in _SEED_NAMES],
        )
        db.commit()


def list_names() -> list[str]:
    rows = _conn().execute("SELECT name FROM associate_roster ORDER BY name COLLATE NOCASE").fetchall()
    return [r["name"] for r in rows]


def add_name(name: str) -> bool:
    name = (name or "").strip()
    if not name:
        return False
    db = _conn()
    db.execute("INSERT OR IGNORE INTO associate_roster (name, added_at) VALUES (?,?)",
               (name, datetime.utcnow().isoformat()))
    db.commit()
    return True


def remove_name(name: str) -> None:
    db = _conn()
    db.execute("DELETE FROM associate_roster WHERE name=?", (name,))
    db.commit()
