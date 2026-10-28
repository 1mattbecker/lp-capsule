# /// script
# requires-python = ">=3.10"
# dependencies = ["codeocean>=0.16,<0.17", "pyyaml>=6"]  # 0.17 needs CO server >=4.8
# ///
"""Launch lp-capsule runs on Code Ocean from a batch config, and collect results.

``predict`` takes a predict config whose ``model`` and ``videos[].asset`` are
data asset names or ids. It starts one worker run per ``--sessions-per-run``
session assets, each with its own assets attached (nothing is committed to the
capsule), keeps at most ``--max-concurrent`` runs of the capsule active,
downloads each run's predictions to ``--out``, and optionally captures each
run as an external result asset in scratch S3.

``train`` takes a train config, runs it, and always captures the result, so
the trained model is a data asset ``predict`` can name.

A run counts as done only when Code Ocean says it succeeded, its exit code is
0, and its ``run.json`` says ``status: ok``. ``<out>/manifest.json`` is
rewritten every pass; launching again with the same ``--out`` skips finished
runs and follows the ones still running.

    uv run scripts/launch.py predict examples/predict_batch.yaml --out runs/r01 --dry-run
    uv run scripts/launch.py train examples/train.yaml --name pylon2test2_ft_r01

The token is ``API_SECRET``, else the macOS Keychain item ``codeocean-api``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codeocean import CodeOcean
from codeocean.computation import (
    ComputationEndStatus,
    ComputationState,
    DataAssetsRunParam,
    NamedRunParam,
    RunParams,
)
from codeocean.data_asset import (
    AWSS3Target,
    ComputationSource,
    DataAsset,
    DataAssetParams,
    DataAssetSearchParams,
    DataAssetState,
    Source,
    Target,
)
from urllib3.util import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from run_config import (
    PredictConfig,
    RunConfig,
    TrainConfig,
    assets_of,
    base_config_asset,
    dump,
    load_config_text,
    parse_config,
)

CO_DOMAIN = "https://codeocean.allenneuraldynamics.org"
# The lp-capsule worker (CO capsule 9589758); --capsule-id overrides.
CAPSULE_ID = "9bc8aec6-aaf9-47b9-93e5-cfad9b8d1a78"
SCRATCH_BUCKET = "aind-scratch-data"
SCRATCH_PREFIX = "matt.becker/lp"
RESULT_TAGS = ["lightning-pose", "lp-capsule"]
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

DEFAULT_MAX_CONCURRENT = 10
POLL_SECONDS = 60
# Consecutive API errors on one run before giving up on it.
MAX_ERRORS = 10
ACTIVE_STATES = {
    ComputationState.Initializing,
    ComputationState.Running,
    ComputationState.Finalizing,
}
IN_FLIGHT = ("running", "capturing")
# Result files downloaded for every run.
SMALL_RESULTS = ("run.json", "run_config.yaml", "pixel_error_summary.csv", "diagnostics.json")
# Folders whose CSVs are downloaded too.
RESULT_FOLDERS = ("predictions", "labeled_frames")


def make_client() -> CodeOcean:
    """Return a Code Ocean client; retries idempotent requests only, never a launch."""
    token = os.environ.get("API_SECRET")
    if not token:
        token = subprocess.run(
            ["security", "find-generic-password", "-s", "codeocean-api", "-w"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    if not token:
        raise RuntimeError("no token: set API_SECRET or the Keychain item codeocean-api")
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    return CodeOcean(domain=CO_DOMAIN, token=token, retries=retries)


def assets_named(client: CodeOcean, name: str) -> list[DataAsset]:
    """Return the unarchived data assets with exactly this name."""
    params = DataAssetSearchParams(query=f"name:{name}", limit=100, archived=False)
    return [a for a in client.data_assets.search_data_assets(params).results if a.name == name]


def resolve_asset(client: CodeOcean, ref: str) -> DataAsset:
    """Return the data asset an id or an exact name refers to."""
    if UUID_RE.fullmatch(ref):
        return client.data_assets.get_data_asset(ref)
    exact = assets_named(client, ref)
    if len(exact) != 1:
        ids = ", ".join(a.id for a in exact) or "none"
        raise LookupError(f"expected one data asset named {ref!r}, found {len(exact)}: {ids}")
    return exact[0]


def resolve_config(
    client: CodeOcean, config: RunConfig, cache: dict[str, DataAsset]
) -> tuple[RunConfig, list[DataAsset]]:
    """Return the config with asset ids replaced by names, and its assets in order."""

    def asset(ref: str) -> DataAsset:
        if ref not in cache:
            cache[ref] = resolve_asset(client, ref)
            print(f"  asset {ref} -> {cache[ref].name} ({cache[ref].id})", flush=True)
        return cache[ref]

    if isinstance(config, PredictConfig):
        resolved: RunConfig = replace(
            config,
            model=asset(config.model).name,
            videos=tuple(replace(v, asset=asset(v.asset).name) for v in config.videos),
        )
    else:
        base = config.base_config
        rest = base[len(base_config_asset(base)) :]
        resolved = replace(
            config,
            project=asset(config.project).name,
            base_config=asset(base_config_asset(base)).name + rest,
            init_checkpoint=(
                asset(config.init_checkpoint).name if config.init_checkpoint else None
            ),
        )
    by_name = {a.name: a for a in cache.values()}
    return resolved, [by_name[n] for n in assets_of(resolved)]


def split_predict(config: PredictConfig, sessions_per_run: int) -> list[PredictConfig]:
    """Split a predict config into runs of ``sessions_per_run`` session assets each.

    A config with no videos (a labeled-frames check alone) is one run; the
    labeled-frames check runs in the first run only.
    """
    sessions = list(dict.fromkeys(v.asset for v in config.videos))
    if not sessions:
        return [config]
    runs = []
    for i in range(0, len(sessions), sessions_per_run):
        group = set(sessions[i : i + sessions_per_run])
        videos = tuple(v for v in config.videos if v.asset in group)
        runs.append(replace(config, videos=videos, labeled_frames=config.labeled_frames and i == 0))
    return runs


def run_label(config: RunConfig, name: str | None = None) -> str:
    """Return a run's name in the manifest and the output folder."""
    if name:
        return name
    assert isinstance(config, PredictConfig)
    sessions = list(dict.fromkeys(v.asset for v in config.videos))
    if not sessions:
        return f"{config.model}_labeled_frames"
    return sessions[0] + (f"+{len(sessions) - 1}" if len(sessions) > 1 else "")


def launch(client: CodeOcean, capsule_id: str, record: dict[str, Any]) -> None:
    """Start one worker run with its assets attached at their names."""
    params = RunParams(
        capsule_id=capsule_id,
        data_assets=[DataAssetsRunParam(id=a["id"], mount=a["name"]) for a in record["assets"]],
        named_parameters=[NamedRunParam(param_name="config", value=record["config"])],
    )
    comp = client.computations.run_capsule(params)
    record.update(computation_id=comp.id, status="running", error=None)
    print(f"  ▸ {record['label']}: computation {comp.id}", flush=True)


def fetch(url: str, path: Path) -> None:
    """Download a URL to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)


def download_results(client: CodeOcean, comp_id: str, out_dir: Path) -> dict[str, Any]:
    """Download a run's small results; return its ``run.json``."""
    computations = client.computations
    top = {item.name: item.path for item in computations.list_computation_results(comp_id).items}
    for name in SMALL_RESULTS:
        if name in top:
            url = computations.get_result_file_download_url(comp_id, top[name]).url
            fetch(url, out_dir / name)
    for folder in RESULT_FOLDERS:
        if folder not in top:
            continue
        for item in computations.list_computation_results(comp_id, top[folder]).items:
            if item.name.endswith(".csv"):
                url = computations.get_result_file_download_url(comp_id, item.path).url
                fetch(url, out_dir / folder / item.name)
    run_json = out_dir / "run.json"
    return json.loads(run_json.read_text()) if run_json.is_file() else {}


def start_capture(client: CodeOcean, record: dict[str, Any], kind: str) -> DataAsset:
    """Capture a finished run as an external data asset in scratch S3."""
    name = record.get("capture_name") or (
        f"{record['label']}_lp_{datetime.now(UTC):%Y-%m-%d_%H-%M-%S}"
    )
    params = DataAssetParams(
        name=name,
        tags=RESULT_TAGS + [kind],
        mount=name,
        source=Source(computation=ComputationSource(id=record["computation_id"])),
        target=Target(
            aws=AWSS3Target(bucket=SCRATCH_BUCKET, prefix=f"{SCRATCH_PREFIX}/{kind}/{name}/")
        ),
    )
    return client.data_assets.create_data_asset(params)


def advance(client: CodeOcean, record: dict[str, Any], out: Path, capture: str | None) -> None:
    """Move one in-flight run on: finished -> checked and downloaded -> captured."""
    label = record["label"]
    if record["status"] == "running":
        comp = client.computations.get_computation(record["computation_id"])
        if comp.state not in (ComputationState.Completed, ComputationState.Failed):
            return
        record["exit_code"] = comp.exit_code
        run = download_results(client, comp.id, out / label) if comp.has_results else {}
        record["run_status"] = run.get("status")
        ok = (
            comp.end_status == ComputationEndStatus.Succeeded
            and comp.exit_code in (0, None)
            and run.get("status") == "ok"
        )
        if not ok:
            end = comp.end_status.value if comp.end_status else "unknown"
            record.update(status="failed", error=run.get("error") or f"end_status {end}")
            print(f"  ✗ {label}: {record['error']}", flush=True)
            return
        if capture is None:
            record["status"] = "completed"
            print(f"  ✓ {label}: results in {out / label}", flush=True)
            return
        asset = start_capture(client, record, capture)
        record.update(result_asset_id=asset.id, result_asset_name=asset.name, status="capturing")
        print(f"  ⇣ {label}: capturing {asset.name}", flush=True)
    elif record["status"] == "capturing":
        asset = client.data_assets.get_data_asset(record["result_asset_id"])
        if asset.state == DataAssetState.Ready:
            record["status"] = "completed"
            print(f"  ✓ {label}: asset {asset.name} ({asset.id})", flush=True)
        elif asset.state == DataAssetState.Failed:
            record.update(status="failed", error=f"capture of {asset.id} failed")
            print(f"  ✗ {label}: {record['error']}", flush=True)


def note_error(record: dict[str, Any], error: Exception, final_status: str) -> None:
    """Count a consecutive API error on a run; give up after ``MAX_ERRORS``."""
    record["_errors"] = record.get("_errors", 0) + 1
    print(f"  ! {record['label']}: {type(error).__name__}: {error}", flush=True)
    if record["_errors"] >= MAX_ERRORS:
        record.update(status=final_status, error=f"{type(error).__name__}: {error}")


def write_manifest(out: Path, manifest: list[dict[str, Any]]) -> None:
    """Write the manifest, without transient ``_`` fields."""
    out.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in manifest]
    tmp = out / "manifest.json.partial"
    tmp.write_text(json.dumps(clean, indent=2) + "\n")
    tmp.replace(out / "manifest.json")


def run_queue(
    client: CodeOcean,
    capsule_id: str,
    manifest: list[dict[str, Any]],
    out: Path,
    max_concurrent: int,
    capture: str | None,
) -> None:
    """Launch queued runs under the cap and follow every run to an end."""
    queue = [r for r in manifest if r["status"] == "queued"]
    while True:
        for record in manifest:
            if record["status"] in IN_FLIGHT:
                try:
                    advance(client, record, out, capture)
                    record["_errors"] = 0
                except Exception as e:  # noqa: BLE001 -- retried next pass
                    note_error(record, e, "error")
        if queue:
            try:
                runs = client.capsules.list_computations(capsule_id)
                free = max_concurrent - sum(1 for c in runs if c.state in ACTIVE_STATES)
            except Exception as e:  # noqa: BLE001 -- retried next pass
                print(f"  ! counting active runs: {type(e).__name__}: {e}", flush=True)
                free = 0
            while free > 0 and queue:
                record = queue[0]
                try:
                    launch(client, capsule_id, record)
                    record["_errors"] = 0
                    queue.pop(0)
                    free -= 1
                except Exception as e:  # noqa: BLE001 -- retried next pass
                    note_error(record, e, "launch_failed")
                    if record["status"] == "launch_failed":
                        queue.pop(0)
                    break
        write_manifest(out, manifest)
        in_flight = sum(1 for r in manifest if r["status"] in IN_FLIGHT)
        if not queue and not in_flight:
            return
        print(
            f"[{datetime.now(UTC):%H:%M}Z] queued={len(queue)} in_flight={in_flight}",
            flush=True,
        )
        time.sleep(POLL_SECONDS)


def plan(
    client: CodeOcean, configs: list[RunConfig], names: list[str | None]
) -> list[dict[str, Any]]:
    """Resolve every run's assets; return new manifest records."""
    cache: dict[str, DataAsset] = {}
    records = []
    for config, name in zip(configs, names):
        resolved, assets = resolve_config(client, config, cache)
        records.append(
            {
                "label": run_label(resolved, name),
                "status": "queued",
                "config": dump(resolved),
                "assets": [{"id": a.id, "name": a.name} for a in assets],
                "computation_id": None,
                **({"capture_name": name} if name else {}),
            }
        )
    return records


def merge_manifest(out: Path, records: list[dict[str, Any]], retry: bool) -> list[dict[str, Any]]:
    """Keep finished and in-flight runs from an earlier launch into ``out``."""
    path = out / "manifest.json"
    if not path.is_file():
        return records
    old = {r["label"]: r for r in json.loads(path.read_text())}
    merged = []
    for record in records:
        prior = old.get(record["label"])
        keep = prior is not None and (
            prior["status"] in ("completed", *IN_FLIGHT)
            or (not retry and prior["status"] not in ("queued", "dry_run"))
        )
        if keep and prior["config"] != record["config"]:
            raise ValueError(
                f"{record['label']}: its config changed since the last launch into {out}; "
                "use a new --out"
            )
        merged.append(prior if keep else record)
    return merged


def main() -> None:
    """Parse arguments, plan the runs, and launch or dry-run them."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="predict a batch of videos")
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, required=True, help="local folder for results")
    p.add_argument("--sessions-per-run", type=int, default=1)
    p.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    p.add_argument("--capture", action="store_true", help="also capture each run as a data asset")
    t = sub.add_parser("train", help="train one model and capture it as a data asset")
    t.add_argument("config", type=Path)
    t.add_argument("--name", required=True, help="name of the model data asset")
    t.add_argument("--out", type=Path, help="local folder (default runs/train/<name>)")
    for s in (p, t):
        s.add_argument("--capsule-id", default=CAPSULE_ID)
        s.add_argument("--dry-run", action="store_true", help="resolve and print only")
        s.add_argument("--retry-failed", action="store_true", help="relaunch failed runs")
    args = parser.parse_args()

    config = parse_config(load_config_text(args.config.read_text()))
    if args.command == "predict":
        if not isinstance(config, PredictConfig):
            parser.error(f"{args.config} is not a predict config")
        if args.sessions_per_run < 1 or args.max_concurrent < 1:
            parser.error("--sessions-per-run and --max-concurrent must be at least 1")
        configs: list[RunConfig] = list(split_predict(config, args.sessions_per_run))
        names: list[str | None] = [None] * len(configs)
        out, max_concurrent = args.out, args.max_concurrent
        capture = "predictions" if args.capture else None
    else:
        if not isinstance(config, TrainConfig):
            parser.error(f"{args.config} is not a train config")
        configs, names = [config], [args.name]
        out, max_concurrent = args.out or Path("runs/train") / args.name, 1
        capture = "models"
    if not args.capsule_id and not args.dry_run:
        parser.error("set CAPSULE_ID in scripts/launch.py or pass --capsule-id")

    client = make_client()
    try:
        manifest = merge_manifest(out, plan(client, configs, names), args.retry_failed)
    except (LookupError, ValueError) as e:
        parser.exit(1, f"error: {e}\n")
    if args.dry_run:
        for record in manifest:
            print(f"\n# {record['label']} [{record['status']}]")
            print("# assets: " + ", ".join(a["name"] for a in record["assets"]))
            print(record["config"], end="")
        return
    for record in manifest:
        if record["status"] not in ("queued", *IN_FLIGHT):
            print(f"  = {record['label']}: {record['status']}, skipped")
    if args.command == "train" and manifest[0]["status"] == "queued":
        # Predict configs name models by asset name, so it must stay unique.
        taken = assets_named(client, args.name)
        if taken:
            ids = ", ".join(a.id for a in taken)
            parser.error(f"a data asset named {args.name} exists ({ids})")
    run_queue(client, args.capsule_id, manifest, out, max_concurrent, capture)
    done = sum(1 for r in manifest if r["status"] == "completed")
    print(f"\n{done}/{len(manifest)} completed; manifest in {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
