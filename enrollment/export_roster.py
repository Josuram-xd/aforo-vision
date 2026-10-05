"""Export the enrolled students as roster.json for aforo-db, without any biometrics.

    python -m enrollment.export_roster [--output data/roster.json]

Reads only the <personId>.json metadata files in data/embeddings/ (never the .npy embeddings) and writes
a list of {"personId", "name"}, the format aforo-db/scripts/seed_people.py loads. Nothing else is copied,
so the file carries no embeddings and no enrollment details.
"""

import argparse
import json
import uuid
from pathlib import Path

from enrollment.enroll_student import EMBEDDINGS_DIR

DEFAULT_OUTPUT = Path("data/roster.json")  # inside data/, which is git-ignored


class RosterError(ValueError):
    pass


def build_roster(embeddings_dir: Path = EMBEDDINGS_DIR) -> tuple[list[dict[str, str]], list[str]]:
    """Return (roster sorted by name, warnings). Raises RosterError on a metadata file that cannot be trusted."""
    roster: list[dict[str, str]] = []
    warnings: list[str] = []
    for path in sorted(embeddings_dir.glob("*.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RosterError(f"{path.name}: invalid JSON ({error})") from None
        person_id = metadata.get("personId") if isinstance(metadata, dict) else None
        name = metadata.get("name") if isinstance(metadata, dict) else None
        if not isinstance(person_id, str) or not _is_uuid_v4(person_id):
            raise RosterError(f"{path.name}: personId must be a UUID v4")
        if person_id != path.stem:
            raise RosterError(f"{path.name}: personId {person_id} does not match the file name")
        if not isinstance(name, str) or not name.strip():
            raise RosterError(f"{path.name}: missing name")
        if not path.with_suffix(".npy").is_file():
            warnings.append(f"{name.strip()} ({person_id}): no embedding file, they would never be recognized; skipped")
            continue
        roster.append({"personId": person_id, "name": name.strip()})

    known = {entry["personId"] for entry in roster} | {p.stem for p in embeddings_dir.glob("*.json")}
    for orphan in sorted(embeddings_dir.glob("*.npy")):
        if orphan.stem not in known:
            warnings.append(f"{orphan.name}: embedding without metadata, not in the roster")

    names = [entry["name"].casefold() for entry in roster]
    for name in sorted({n for n in names if names.count(n) > 1}):
        warnings.append(f"repeated name '{name}': check that they are two different people")
    roster.sort(key=lambda entry: entry["name"].casefold())
    return roster, warnings


def _is_uuid_v4(value: str) -> bool:
    try:
        return uuid.UUID(value).version == 4
    except ValueError:
        return False


def write_roster(roster: list[dict[str, str]], output: Path = DEFAULT_OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(roster, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Exporta roster.json (personId + name, sin biometría) para aforo-db.")
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        roster, warnings = build_roster(args.embeddings_dir)
    except RosterError as error:
        print(f"No se exportó nada: {error}")
        return 1
    for warning in warnings:
        print(f"Aviso: {warning}")
    if not roster:
        print(f"No hay personas enroladas en {args.embeddings_dir}; no se escribió el roster.")
        return 1
    print(f"Listo: {write_roster(roster, args.output)} ({len(roster)} personas). Cópialo a aforo-db/scripts/roster.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
