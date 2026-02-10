from __future__ import annotations

from dataclasses import dataclass
import argparse
import glob
import importlib
from pathlib import Path
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    artifact_glob: str
    import_name: str
    variable_name: str


def _parse_args(*, argv: Sequence[str]) -> SmokeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Install a built Python package artifact and run a minimal import/runtime smoke test."
        )
    )
    parser.add_argument(
        "--artifact-glob",
        required=True,
        help="Glob pattern for built distributions, for example 'dist/*.whl'.",
    )
    parser.add_argument(
        "--import-name",
        default="arco",
        help="Top-level import to validate after installation.",
    )
    parser.add_argument(
        "--variable-name",
        default="x",
        help="Variable name used in the model construction smoke check.",
    )

    args = parser.parse_args(argv)
    return SmokeConfig(
        artifact_glob=args.artifact_glob,
        import_name=args.import_name,
        variable_name=args.variable_name,
    )


def _resolve_artifacts(*, pattern: str) -> list[Path]:
    artifacts = [Path(path).resolve() for path in glob.glob(pattern)]
    if not artifacts:
        raise FileNotFoundError(f"No package artifacts matched pattern: {pattern}")
    return sorted(artifacts)


def _install_artifacts(*, artifacts: Sequence[Path]) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", *[str(path) for path in artifacts]]
    )


def _run_import_and_model_smoke(*, import_name: str, variable_name: str) -> None:
    module = importlib.import_module(import_name)
    model = module.Model()
    model.add_variable(bounds=module.Bounds(lower=0.0, upper=1.0), name=variable_name)


def main(*, argv: Sequence[str]) -> int:
    config = _parse_args(argv=argv)
    artifacts = _resolve_artifacts(pattern=config.artifact_glob)
    _install_artifacts(artifacts=artifacts)
    _run_import_and_model_smoke(
        import_name=config.import_name,
        variable_name=config.variable_name,
    )
    print(f"smoke-ok import={config.import_name} artifacts={len(artifacts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv=sys.argv[1:]))
