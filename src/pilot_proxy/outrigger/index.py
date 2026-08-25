"""Build a resumable N2 header index."""
from __future__ import annotations

from pathlib import Path

from pilot_proxy.archive.header_index import inspect_headers
from pilot_proxy.archive.instruments import load_instrument
from pilot_proxy.archive.interfaces import RunContext
from pilot_proxy.archive.sources import CadcDatatrailSource, LocalDirectorySource

from .n2_reader import OutriggerN2Reader


def build_n2_index(
    *,
    output: str | Path,
    input_dir: str | Path | None = None,
    inventory: str | Path | None = None,
    scratch: str | Path | None = None,
    site: str = "gbo",
    source_glob: str = "*.h5",
    allow_partial: bool = False,
) -> Path:
    """Inspect local or inventoried N2 files and write their time coverage."""
    if (input_dir is None) == (inventory is None):
        raise SystemExit("n2-index needs exactly one of --input-dir or --inventory")
    output_path = Path(output).expanduser()
    scratch_path = (
        Path(scratch).expanduser()
        if scratch is not None
        else output_path.parent / "_n2_staging"
    )
    options: dict[str, object]
    if input_dir is not None:
        source = LocalDirectorySource()
        options = {
            "source_root": str(Path(input_dir).expanduser()),
            "source_glob": source_glob,
        }
    else:
        source = CadcDatatrailSource()
        options = {"inventory": str(Path(inventory).expanduser())}

    ctx = RunContext(instrument=load_instrument(site), options=options)
    reader = OutriggerN2Reader()
    source_check = (
        source.fetch_preflight
        if isinstance(source, CadcDatatrailSource)
        else source.preflight
    )
    source_result = source_check(ctx)
    ok, problems = bool(source_result[0]), list(source_result[1])
    if not ok:
        raise SystemExit("n2-index source: " + "; ".join(problems))
    ok, problems = reader.preflight(ctx)
    if not ok:
        raise SystemExit("n2-index reader: " + "; ".join(problems))
    units = list(source.enumerate(ctx))
    result = inspect_headers(
        source=source,
        reader=reader,
        units=units,
        ctx=ctx,
        output=output_path,
        scratch=scratch_path,
    )
    print(
        f"N2 index: {result.inspected} inspected, {result.cached} cached, "
        f"{result.failed} failed -> {output_path}"
    )
    if result.failed and not allow_partial:
        raise SystemExit("n2-index is incomplete; rerun or pass --allow-partial")
    return output_path
