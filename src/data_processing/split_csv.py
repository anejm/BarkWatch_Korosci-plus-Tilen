"""
Split large CSV files into <100MB parts for GitHub upload, and recombine them.
Works at the raw byte level — no parsing, fully lossless.

Usage:
    split("data/processed/odseki.csv")
    combine("data/processed/odseki_01.csv")  # pass any part, finds the rest automatically
"""

import re
from pathlib import Path

MAX_BYTES = 95_000_000  # 95 MB decimal — safely under GitHub's 100 MB limit


def split(file_path: str | Path, max_bytes: int = MAX_BYTES) -> list[Path]:
    file_path = Path(file_path)
    file_size = file_path.stat().st_size

    if file_size <= max_bytes:
        print(f"{file_path.name} is {file_size / 1e6:.1f} MB — no splitting needed.")
        return [file_path]

    stem = file_path.stem
    suffix = file_path.suffix
    out_dir = file_path.parent
    parts = []

    with open(file_path, "rb") as f:
        header = f.readline()
        part_idx = 1
        current_lines = [header]
        current_size = len(header)

        for raw_line in f:
            # If adding this line would exceed the limit, flush current chunk first
            if current_size + len(raw_line) > max_bytes and len(current_lines) > 1:
                out_path = out_dir / f"{stem}_{part_idx:02d}{suffix}"
                out_path.write_bytes(b"".join(current_lines))
                size_mb = out_path.stat().st_size / 1e6
                print(f"  Written {out_path.name} ({len(current_lines) - 1} rows, {size_mb:.1f} MB)")
                parts.append(out_path)
                part_idx += 1
                current_lines = [header]
                current_size = len(header)

            current_lines.append(raw_line)
            current_size += len(raw_line)

        # Write final chunk
        if len(current_lines) > 1:
            out_path = out_dir / f"{stem}_{part_idx:02d}{suffix}"
            out_path.write_bytes(b"".join(current_lines))
            size_mb = out_path.stat().st_size / 1e6
            print(f"  Written {out_path.name} ({len(current_lines) - 1} rows, {size_mb:.1f} MB)")
            parts.append(out_path)

    print(f"Split {file_path.name} into {len(parts)} parts.")
    return parts


def combine(any_part: str | Path, out_path: str | Path | None = None) -> Path:
    any_part = Path(any_part)

    match = re.match(r"^(.+?)_(\d+)$", any_part.stem)
    if not match:
        raise ValueError(f"Cannot detect part pattern from filename: {any_part.name}")

    base_stem = match.group(1)
    suffix = any_part.suffix
    directory = any_part.parent

    parts = sorted(directory.glob(f"{base_stem}_*{suffix}"))
    parts = [p for p in parts if re.match(rf"^{re.escape(base_stem)}_\d+$", p.stem)]

    if not parts:
        raise FileNotFoundError(f"No parts found for pattern '{base_stem}_NN{suffix}' in {directory}")

    print(f"Found {len(parts)} parts: {[p.name for p in parts]}")

    if out_path is None:
        out_path = directory / f"{base_stem}{suffix}"
    out_path = Path(out_path)

    with open(out_path, "wb") as out:
        for i, part in enumerate(parts):
            with open(part, "rb") as f:
                header = f.readline()
                if i == 0:
                    out.write(header)  # write header only from first part
                out.write(f.read())

    size_mb = out_path.stat().st_size / 1e6
    print(f"Combined into {out_path.name} ({size_mb:.1f} MB)")
    return out_path


if __name__ == "__main__":
    parts = split("data/processed/odseki.csv")
    if len(parts) > 1:
        combine(parts[0])
