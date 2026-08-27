"""
storage/
Persistent object-storage abstraction (Cloudflare R2, S3-compatible), used
only when STORAGE_BACKEND=r2. See storage/r2.py, storage/persistent_file.py.

Independent of DB_BACKEND — this package has no import or runtime
dependency on config/settings.py's database settings, and vice versa.
"""
