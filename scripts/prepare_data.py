import toml
import sqlite3
import subprocess
from pathlib import Path


def prepare_new_database(source_database, ml_database):
    """
    Copy the database into a new file.
    """

    with sqlite3.connect(source_database) as src:
        with sqlite3.connect(ml_database) as dst:
            src.backup(dst)

    create_new_tables = Path("schema/additional_tables.sql").read_text()
    # Add a new table to the database in which to store split data
    with sqlite3.connect(ml_database) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.executescript(create_new_tables)
        conn.commit()


def create_ml_view(ml_database, view_file):
    """
    Create a view which extracts an ML-ready subset
    """
    surfpro_view = view_file.read_text()
    with sqlite3.connect(ml_database) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.execute(surfpro_view)
        conn.commit()


def main():
    config = toml.loads(Path("config.toml").read_text())

    REPO_URL = config["REPOSITORY_URL"]
    CLONE_DIR = Path(config["CLONE_DIR"])
    SOURCE_DATABASE = Path(config["DB_PATH"])
    ML_DATABASE = Path(config["ML_DATABASE"])
    VIEW_FILE = Path(config["ML_VIEW_QUERY"])

    # Command to run inside the cloned repo
    COMMAND = ["python", "scripts/initialise_database.py"]

    def run(*args, cwd=None):
        subprocess.run(args, cwd=cwd, check=True)

    # Clone if missing
    if not (CLONE_DIR / ".git").exists():
        CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", REPO_URL, str(CLONE_DIR))

    # Ensure .gitignore contains the folder
    gitignore = Path(".gitignore")
    entry = CLONE_DIR.as_posix()

    existing = gitignore.read_text().splitlines() if gitignore.exists() else []
    if entry not in existing:
        with gitignore.open("a", newline="\n") as f:
            if existing:
                f.write("\n")
            f.write(f"{entry}\n")

    # Run the target command
    subprocess.run(COMMAND, cwd=CLONE_DIR, check=True)

    prepare_new_database(SOURCE_DATABASE, ML_DATABASE)

    create_ml_view(ML_DATABASE, VIEW_FILE)


if __name__ == "__main__":
    main()
