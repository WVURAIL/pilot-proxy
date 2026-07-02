# coding=utf-8
"""Drive the datatrawl streaming engine over CHIME data, then combine.

This is the implementation behind ``pilot-proxy chime-scan``. It reuses datatrawl's
engine and plugin registry exactly as datatrawl's own ``scan`` command does
(``registry.get`` -> ``plan_runs`` fan-out -> per-channel ``pipeline.run``), so a
multi-channel pull is storage-safe and resumable, then stacks the per-pilot
products with the combine step into PilotProxy's canonical products. This is the
recommended archive-scale entry point; ``pilot-proxy chime-run`` (the
``run_chime_analysis`` batch path) remains for pre-staged local directories.

* ``--source local``          : files already on disk (a 10 s chunk, /arc, ...).
* ``--source cadc-datatrail``  : storage-safe streaming from the CADC archive.

The detector analyzer defaults to the real CUDA kernel (GPU). For tests, the
detector / kernel / weights can be injected via ``analyzer_options`` (the same
hooks ``run_chime_analysis`` exposes), which is how the GPU-free parity tests run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

_DETECTOR_ANALYZER = "pilot-proxy-detector"
_READER_FOR_ANALYZER = {
    _DETECTOR_ANALYZER: "chime-baseband-packed",  # native int4 -> lossless kernel pack
}


def _named_inventory_path(name: str, source_root: str | Path | None = None) -> Path:
    """Return ``<root>/data/<name>/inventory.jsonl`` for a named datatrawl inventory.

    ``source_root`` is the directory passed to ``datatrawl survey --root``. When it
    is omitted, datatrawl and pilot-proxy both interpret the current working
    directory as the root. This keeps
    ``datatrawl survey --freq-ids <pilots> --name chime-pilots`` followed by
    ``pilot-proxy chime-scan --inventory-name chime-pilots`` deterministic as long as
    both commands are run from the same directory.
    """
    if not str(name).strip():
        raise ValueError("inventory name may not be empty")
    root = Path.cwd() if source_root is None else Path(source_root)
    return root / "data" / str(name).strip() / "inventory.jsonl"


def run_chime_scan(
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path,
    source: str = "local",
    analyzer: str = _DETECTOR_ANALYZER,
    select: Any = None,
    instrument: str = "chime",
    reader: str | None = None,
    max_files: int | None = None,
    max_chunks_per_file: int | None = None,
    work_dir: str | Path | None = None,
    source_glob: str = "*.h5",
    source_channel_regex: str | None = None,
    inventory: str | Path | None = None,
    inventory_name: str | None = None,
    source_root: str | Path | None = None,
    download_workers: int = 1,
    max_staged_files: int = 1,
    checkpoint_every: int | None = None,
    analyzer_options: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Fan out the chosen analyzer over CHIME data and combine into canonical products.

    Source plumbing is explicit per source: ``--source local`` reads files under
    ``--input-dir``. ``--source cadc-datatrail`` streams from an inventory, provided
    as one of:

    * ``--inventory <inventory.jsonl>`` for an explicit inventory path;
    * ``--inventory-name <name>`` for newer datatrawl named inventories, resolved as
      ``<root>/data/<name>/inventory.jsonl`` where ``root`` is ``--source-root`` or
      the current working directory;
    * ``--source-root <dir>`` alone for the legacy
      ``<root>/data/<instrument>/inventory.jsonl`` layout.

    The detector analyzer's CUDA kernel is GPU-only, so ``--analyzer pilot-proxy-detector``
    requires a GPU node -- a missing ``cupy`` is caught up front (rather than
    surfacing as every file failing to analyze). There is no CPU detector path for
    production (the CPU reference exists only as a test fixture), so there is no
    GPU/CPU toggle here.
    """
    from datatrawl import pipeline, registry
    from datatrawl.instruments import load_instrument
    from datatrawl.interfaces import RunContext

    from .combine import combine_detector_products

    registry.load_plugins()  # bundled datatrawl plugins + pilot-proxy's entry-point plugins
    if analyzer != _DETECTOR_ANALYZER:
        raise SystemExit(
            f"chime-scan: unknown analyzer {analyzer!r} "
            f"(expected {_DETECTOR_ANALYZER!r})"
        )
    reader_name = reader or _READER_FOR_ANALYZER[analyzer]

    if select is None or (isinstance(select, str) and not select.strip()) or (
        isinstance(select, (list, tuple)) and len(select) == 0
    ):
        raise SystemExit(
            "chime-scan: --select is required (e.g. --select 844 or --select 829,844). "
            "It names CHIME freq_id coarse channels (one freq_id = one pilot). The "
            "PilotProxy analyzers have no 'all' mode, to avoid accumulating multiple "
            "channels into one product."
        )

    # The PilotProxy analyzers append frames in delivery order; with download_workers > 1
    # or max_staged_files > 1, datatrawl may deliver files out of source order, which
    # would corrupt frame_index / relative_time_s. Force the single-file, order-safe
    # path regardless of caller request.
    if (int(download_workers), int(max_staged_files)) != (1, 1):
        print(
            "[chime-scan] note: forcing download_workers=1, max_staged_files=1; the "
            "PilotProxy analyzers are order-sensitive and require datatrawl's single-file path.",
            flush=True,
        )
        download_workers = 1
        max_staged_files = 1

    inst = load_instrument(instrument)
    options: dict[str, Any] = dict(analyzer_options or {})
    if source == "local":
        if input_dir is None and source_root is None:
            raise SystemExit(
                "chime-scan: --source local needs --input-dir <dir> (the directory "
                "of baseband_<event>_<freq_id>.h5 files)."
            )
        options["source_root"] = str(input_dir if input_dir is not None else source_root)
        options["source_glob"] = source_glob
        if source_channel_regex:
            options["source_channel_regex"] = source_channel_regex
    elif source == "cadc-datatrail":
        # datatrawl's CADC source reads ctx.options["inventory"] (explicit path) or
        # ctx.options["root"] (the legacy <root>/data/<instrument>/inventory.jsonl
        # layout). Newer datatrawl surveys write named inventories under
        # <root>/data/<name>/, so support that directly here by resolving it to an
        # explicit inventory path before calling the source.
        if inventory is not None and inventory_name is not None:
            raise SystemExit(
                "chime-scan: pass either --inventory <inventory.jsonl> or "
                "--inventory-name <name>, not both."
            )
        if inventory is not None:
            options["inventory"] = str(inventory)
        elif inventory_name is not None:
            options["inventory"] = str(_named_inventory_path(inventory_name, source_root))
        elif source_root is not None:
            options["root"] = str(source_root)
        else:
            raise SystemExit(
                "chime-scan: --source cadc-datatrail needs --inventory "
                "<inventory.jsonl>, --inventory-name <name> for "
                "<survey-root>/data/<name>/inventory.jsonl, or --source-root <dir> "
                "alone for the legacy <dir>/data/<instrument>/inventory.jsonl "
                "layout. Build the inventory with `datatrawl survey` first, and "
                "pass --source-root if the survey root is not the current directory."
            )
    else:
        raise SystemExit(
            f"chime-scan: unknown source {source!r} "
            f"(expected 'local' or 'cadc-datatrail')."
        )
    if max_chunks_per_file is not None:
        options["max_chunks_per_file"] = int(max_chunks_per_file)

    ctx = RunContext(instrument=inst, options=options)

    src = registry.get("source", source)()
    rdr = registry.get("reader", reader_name)()
    analyzer_cls = registry.get("analyzer", analyzer)

    # Fail fast on missing runtime artifacts -- for pilot-proxy-detector that is cupy,
    # the CUDA kernel library, and the weight bank -- before any file is staged,
    # instead of quarantining every unit or dying with a raw error mid-scan.
    _ok, _problems = analyzer_cls().preflight(ctx)
    if not _ok:
        raise SystemExit(
            f"chime-scan: {analyzer} preflight failed:\n  - "
            + "\n  - ".join(_problems)
            + "\n  (a detector run needs a GPU node with a built "
            "cuda/libfstatistic.so and the weight bank -- run setup_env.sh on a "
            "GPU node.)")

    runs = analyzer_cls().plan_runs(ctx, select)
    if not runs:
        raise SystemExit("chime-scan: --select resolved to an empty set")

    work = Path(work_dir) if work_dir is not None else Path(output_dir) / "_per_pilot"
    work.mkdir(parents=True, exist_ok=True)
    tmp_dir = str(work / "_staging")
    quarantine_path = str(work / "quarantine.jsonl")

    product_paths: list[str] = []
    for sub_sel in runs:
        ctx.selection = sub_sel
        units = list(src.enumerate(ctx))
        stem = ("_".join(str(s) for s in sub_sel)
                if isinstance(sub_sel, (list, tuple)) else str(sub_sel))
        if not units:
            if verbose:
                print(f"  [chime-scan] select={sub_sel}: no files matched -- skipping",
                      flush=True)
            continue
        out = str(work / f"{stem}.npz")
        if verbose:
            print(f"  [chime-scan] select={sub_sel}: {len(units)} file(s) -> {out}",
                  flush=True)
        analyzer_obj = analyzer_cls()  # fresh analyzer per product
        result = pipeline.run(
            source=src, reader=rdr, analyzer=analyzer_obj, units=units,
            out_path=out, tmp_dir=tmp_dir, ctx=ctx,
            download_workers=int(download_workers),
            max_staged_files=int(max_staged_files),
            max_files=max_files, max_frames_per_file=max_chunks_per_file,
            checkpoint_every=(50 if checkpoint_every is None else int(checkpoint_every)),
            quarantine_path=quarantine_path, verbose=False,
        )
        # The engine only writes the product if at least one unit was accumulated;
        # if every unit failed/quarantined there is no product (or it has zero
        # frames). Treat that as an error rather than silently feeding an absent/
        # empty product to combine -- it usually signals a systemic problem
        # (missing GPU, a bad inventory) that would hit every channel, not just
        # bad input for this one. Use n_done (total accumulated, this run plus any
        # resumed), not n_new, so a relaunch that finds a channel already complete
        # (n_new == 0) is recognised as produced rather than mistaken for a failure.
        produced = Path(out).exists() and int(getattr(result, "n_done", 0)) > 0
        if not produced:
            raise SystemExit(
                f"chime-scan: freq_id {sub_sel}: no usable product -- "
                f"{int(getattr(result, 'n_failed', 0))} of {len(units)} unit(s) "
                f"failed, {int(getattr(result, 'n_quarantined', 0))} quarantined "
                f"(see {quarantine_path}). For pilot-proxy-detector this is most often a "
                f"missing GPU/cupy environment; run on a GPU node."
            )
        product_paths.append(out)

    if not product_paths:
        raise SystemExit(
            "chime-scan: no products produced -- no files matched the selection "
            f"(source={source}, select={select})"
        )

    outputs = combine_detector_products(product_paths, output_dir)
    if verbose:
        print(f"[chime-scan] combined {len(product_paths)} pilot product(s) -> {output_dir}",
              flush=True)
        for label, path in outputs.items():
            print(f"  {label}: {path}", flush=True)
    return outputs


__all__ = ["run_chime_scan"]
