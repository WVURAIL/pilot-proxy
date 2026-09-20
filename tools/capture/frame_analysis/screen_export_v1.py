#!/usr/bin/env python3
"""Export a scalar screen without promoting it to a scientific channel ruling."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def qualify_row(row):
    channel = int(row['channel'])
    result = {key: row[key] for key in ('freq_id', 'channel', 'role')}
    result.update({'screen_' + key: value for key, value in row.items()
                   if key not in ('freq_id', 'channel', 'role')})
    result.update(scientific_ruling='undetermined' if 14 <= channel <= 36 else 'control',
        physical_recovery_certified=False, physical_exclusion_certified=False,
        scientific_reason=('Frame-policy visibility transfer, confidence bounds, retained science information, '
                           'and the allowed-mask lower bound remain unqualified.' if 14 <= channel <= 36
                           else 'Reference allocation; outside the 23 channel decisions.'))
    return result


def export(source, output):
    if output.exists():
        raise FileExistsError(f'will not overwrite {output}')
    raw = source.read_bytes()
    rows = [qualify_row(row) for row in csv.DictReader(raw.decode().splitlines())]
    if not rows:
        raise ValueError('screen is empty')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    receipt = dict(schema='qualified_capture_screen_v1', source=str(source),
                   source_sha256=hashlib.sha256(raw).hexdigest(), output=str(output),
                   output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                   rows=len(rows), scientific_status='no scientific channel ruling inferred from this screen')
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
