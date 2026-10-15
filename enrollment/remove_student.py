"""Delete a student's biometrics from this machine (right to erasure, Ley 1581).

    python -m enrollment.remove_student --person-id <uuid> [--yes]

Removes data/embeddings/<personId>.npy (the embedding) and <personId>.json (name and enrollment date), and
drops the person from data/roster.json if that file exists. The running pipeline keeps its in-memory index
until it restarts, so restart `python -m src.main` afterwards. Deleting the student in the dashboard
(aforo-frontend, Estudiantes → Eliminar) is a separate step: that one erases the data in Supabase.
"""

import argparse
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from enrollment.enroll_student import EMBEDDINGS_DIR
from enrollment.export_roster import DEFAULT_OUTPUT as ROSTER_PATH


@dataclass
class Removal:
    deleted: list[Path] = field(default_factory=list)
    roster_updated: bool = False


def parse_person_id(value: str) -> str:
    """Canonical UUID v4 string. Rejects anything else, so the id can never become a path like ../x."""
    try:
        parsed = uuid.UUID(value.strip())
    except ValueError:
        raise ValueError(f"personId must be a UUID v4: {value!r}") from None
    if parsed.version != 4 or str(parsed) != value.strip().lower():
        raise ValueError(f"personId must be a UUID v4: {value!r}")
    return str(parsed)


def remove_student(person_id: str, embeddings_dir: Path = EMBEDDINGS_DIR, roster_path: Path = ROSTER_PATH) -> Removal:
    person_id = parse_person_id(person_id)
    result = Removal()
    for path in (embeddings_dir / f"{person_id}.npy", embeddings_dir / f"{person_id}.json"):
        if path.is_file():
            path.unlink()
            result.deleted.append(path)

    if roster_path.is_file():
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        if isinstance(roster, list):
            kept = [entry for entry in roster if not (isinstance(entry, dict) and entry.get("personId") == person_id)]
            if len(kept) != len(roster):
                roster_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                result.roster_updated = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Borra la biometría de un estudiante de este equipo (supresión).")
    parser.add_argument("--person-id", required=True, help="UUID del estudiante (el mismo id que en el dashboard)")
    parser.add_argument("--yes", action="store_true", help="no pedir confirmación")
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--roster", type=Path, default=ROSTER_PATH)
    args = parser.parse_args()

    try:
        person_id = parse_person_id(args.person_id)
    except ValueError as error:
        print(f"No se borró nada: {error}")
        return 1

    if not args.yes:
        answer = input(f"Se borrará la biometría de {person_id} de este equipo. No se puede deshacer. Escribe BORRAR: ")
        if answer.strip() != "BORRAR":
            print("Cancelado: no se borró nada.")
            return 1

    result = remove_student(person_id, args.embeddings_dir, args.roster)
    if not result.deleted and not result.roster_updated:
        print(f"No había datos de {person_id} en este equipo.")
        return 0
    for path in result.deleted:
        print(f"Borrado: {path}")
    if result.roster_updated:
        print(f"Quitado de {args.roster}")
    print("Reinicia el pipeline (python -m src.main) para que deje de reconocerlo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
