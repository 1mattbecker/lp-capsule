"""Lightning Pose predict or train on Code Ocean: one run, one config.

The App Panel's one field, ``config``, is the run config as YAML (or JSON) text,
or a path to a YAML file: absolute, or relative to ``/data`` or ``code/``. See
:mod:`run_config` for its keys. Outputs in ``/results``:

- ``run_config.yaml``: the config as run, defaults filled;
- ``run.json``: versions, model and checkpoint hashes, timings, and
  ``status``, written ``ok`` last (``failed`` with the error otherwise);
- predict: ``predictions/<asset>_<camera>.csv``, LP's CSV format indexed by
  source frame;
- train: ``model/`` (a model bundle) and ``pixel_error_summary.csv``.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import traceback
from pathlib import Path

from record import RunRecord
from run_config import (
    PredictConfig,
    RunConfig,
    assets_of,
    dump,
    load_config_text,
    parse_config,
)

CODE_DIR = Path(__file__).resolve().parent
_log = logging.getLogger("lp_capsule")


def read_config_arg(value: str, data_dir: Path) -> str:
    """Return config text: ``value`` itself, or the file it names."""
    value = value.strip()
    if not value:
        raise ValueError("the config parameter is empty; give YAML text or a path")
    if "\n" not in value and value.endswith((".yaml", ".yml", ".json")):
        for path in (Path(value), data_dir / value, CODE_DIR / value):
            if path.is_file():
                _log.info("config from %s", path)
                return path.read_text()
        raise FileNotFoundError(f"config file {value} not found as given, in /data or code/")
    return value


def check_mounts(config: RunConfig, data_dir: Path) -> None:
    """Fail early, and clearly, if an asset the config names is not mounted."""
    missing = [name for name in assets_of(config) if not (data_dir / name).is_dir()]
    if missing:
        mounted = sorted(p.name for p in data_dir.iterdir()) if data_dir.is_dir() else []
        raise FileNotFoundError(
            f"assets not mounted under {data_dir}: {missing}; mounted: {mounted or 'none'}. "
            "Attach them with these mount names."
        )


def run_task(config: RunConfig, data_dir: Path, results_dir: Path, record: RunRecord) -> None:
    """Run the config's task in a temporary work folder under ``results_dir``."""
    # Under /results, not /scratch: /scratch is EFS and slow. Removed on exit.
    with tempfile.TemporaryDirectory(prefix=".work-", dir=results_dir) as work_dir:
        work = Path(work_dir)
        if isinstance(config, PredictConfig):
            from predict import run_predict

            run_predict(config, data_dir, results_dir, work, record)
        else:
            from train import run_train

            run_train(config, data_dir, results_dir, work, record)


def run(argv: list[str] | None = None) -> None:
    """Parse the App Panel parameter and run its task (the template's top-level run)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="YAML text or a path")
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--results-dir", type=Path, default=Path("/results"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    record = RunRecord(args.results_dir, task="unknown")
    try:
        config = parse_config(load_config_text(read_config_arg(args.config, args.data_dir)))
        record.update(task=config.task, assets=assets_of(config))
        (args.results_dir / "run_config.yaml").write_text(dump(config))
        _log.info("config:\n%s", dump(config))
        check_mounts(config, args.data_dir)
        run_task(config, args.data_dir, args.results_dir, record)
    except BaseException as e:
        record.write("failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        raise
    record.write("ok")


if __name__ == "__main__":
    run()
