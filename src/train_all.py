import subprocess
import sys
import time
from pathlib import Path


def run_and_log(script: str, log_dir: Path, timestamp: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{timestamp}.log"
    command = ["pipenv", "run", "python", script]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    timestamp = time.strftime("%Y%m%d%H%M")
    jobs = [
        ("deepdti.py", Path("logs/deepdti")),
        ("kge_nfm.py", Path("logs/kge_nfm")),
        ("kge_rf.py", Path("logs/kge_rf")),
    ]
    for script, log_dir in jobs:
        run_and_log(script, log_dir, timestamp)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
