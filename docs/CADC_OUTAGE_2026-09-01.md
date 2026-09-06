# Global raven cannot resolve `ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca` — all `cadcget` downloads fail with HTTP 500

**Observed:** 2026-09-01, from ~11:04 UTC to ~23:08 UTC (~12 h)
**Updated:** 2026-09-06, after the affected campaign completed. Sections
marked *(post-run)* were added then and were not part of the original report.
**Affects:** every Storage Inventory download through the default service,
including **anonymous access to public collections**

## Summary

The global raven service (`ivo://cadc.nrc.ca/global/raven`) — the default
locator for `cadcget` and `StorageInventoryClient` — returns **HTTP 500** on
every `POST /raven/locate`, so no downloads succeed.

raven's own VOSI availability endpoint reports the cause: a
**`java.net.UnknownHostException`** when it tries to reach the credential
service at `ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca`.

**That host is healthy.** From an external network it resolves normally, serves
a valid TLS certificate, and `GET /cred/availability` (with a client cert)
returns **HTTP 200** with `<vosi:available>true</vosi:available>`, "service is
accepting requests", in 0.46 s. The dependency is up; raven simply cannot
resolve its hostname. This looks like a **DNS/resolver failure inside the raven
deployment**, not a failure of `cred` itself.

## Reproduction

Authenticated:

```bash
cadcget --cert ~/.ssl/cadcproxy.pem \
  cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/astro_100058001/baseband_100058001_506.h5 \
  -o /tmp/test.h5
# ERROR: unexpected exception: java.lang.RuntimeException:
#        unexpected exception calling permissions service(s)
```

**Anonymous, public collection, no certificate at all** — fails identically,
which rules out any client credential or account issue:

```bash
cadcget cadc:IRIS/I429B4H0.fits -o /tmp/anon.fits
# ERROR: unexpected exception: java.lang.RuntimeException:
#        unexpected exception calling permissions service(s)
```

Verbose (`-d`) shows the failing call:

```
POST https://cadc-west-01.canfar.net/raven/locate HTTP/1.1" 500 101
```

## The diagnostic evidence

raven self-reports the failure:

```bash
curl -s https://cadc-west-01.canfar.net/raven/availability
```

```xml
<vosi:available>false</vosi:available>
<vosi:note>availability check failed:
  https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cred/availability
  code: -1 cause: java.net.UnknownHostException:
  ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca</vosi:note>
```

The named dependency answers fine from outside:

```bash
curl --cert ~/.ssl/cadcproxy.pem --key ~/.ssl/cadcproxy.pem \
  https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cred/availability
# HTTP 200, 0.46 s
# <vosi:available>true</vosi:available>
# <vosi:note>service is accepting requests</vosi:note>
```

DNS and transport to that host are also healthy externally: it resolves to
`132.246.217.29`, TCP connects succeed on 443 and 80, and TLS verifies against
a valid Entrust-issued certificate (`verify return code: 0 (ok)`).

## Which services are involved

Per the registry at `https://cadc-west-01.canfar.net/reg/resource-caps`, both
permission-related dependencies raven needs live on the host it cannot resolve:

```
ivo://cadc.nrc.ca/global/baldur = https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/baldur/capabilities
ivo://cadc.nrc.ca/cred          = https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cred/capabilities
```

The client-facing message names "permissions service(s)" (baldur); raven's
availability note names `cred`. Both resolve to the same unreachable-from-raven
hostname, which is consistent with a single name-resolution failure affecting
every dependency raven has at that site.

## Scope

| Component | State |
|---|---|
| global raven (`cadc-west-01.canfar.net/raven`) | **available: false** — the outage |
| `cred` (Victoria) | available: **true**, HTTP 200 — healthy, but unreachable *from raven* |
| minoc (UVic, `ws-uv.canfar.net`) | healthy, **serves data correctly** |
| minoc (CADC, `ws-cadc.canfar.net`) | available: true |
| luskan | available: true |
| VOSpace (`vls` / `vcp` to `/arc`) | works normally |

Not credential-related (anonymous public access fails identically). Not
collection-specific (`cadc:IRIS/...` and `cadc:CHIMEFRB/...` fail the same way).
Reproduced from two independent networks: a local Linux host and a CANFAR
`astroml-cuda` notebook session.

## Why this may have gone unreported

`GET /minoc/capabilities` and `GET /raven/capabilities` both returned **HTTP 200
throughout**. Monitoring that checks capabilities endpoints — or simple
HTTP reachability — would see nothing wrong. Only `/raven/availability`, which
actually exercises the dependency, reports the failure.

## Workaround (verified byte-identical)

Addressing a minoc replica directly, bypassing raven, works:

```bash
cadcget --cert ~/.ssl/cadcproxy.pem \
  -s ivo://cadc.nrc.ca/uvic/minoc \
  cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/astro_100058001/baseband_100058001_506.h5 \
  -o /tmp/test.h5
```

Returns 91,311,880 bytes, md5 `62441de83c1b4f0f9b734f4264697425` — identical to
the same object fetched via raven before the outage. The URL form
(`-s https://ws-uv.canfar.net/minoc`) works too, as does a ranged HTTP GET
against `/minoc/files/<uri>`.

## Environment

`cadcdata` 2.5.2, `cadcutils` 1.6.2, Python 3.12; X.509 proxy certificate
(valid to 2026-09-28) and anonymous access both tested.

## What followed *(post-run)*

The campaign that hit this ran to completion on 2026-09-06 and gives a fuller
picture. Availability was the dominant constraint throughout: a 26.33 TiB
retrieval that three concurrent sessions can sustain at ~125 MiB/s -- about
60 hours of transfer -- took five days.

**Further interruptions, of a different kind.** After raven recovered, the
byte path failed twice more with a signature unlike the DNS failure above:
`/raven/availability` reported `true` and `cadcinfo` returned full metadata,
while `cadcget` returned nothing.

| when (UTC) | signature |
|---|---|
| 09-01 11:04 – 23:08 | raven `available: false`, `UnknownHostException` for the cred/baldur host |
| 09-02 ~02:25 – 03:11 | raven `true`, `cadcinfo` answers, `cadcget` returns nothing |
| 09-03 ~18:33 | same as above |

**Sustained degradation.** On 2026-09-02 from roughly 04:00 to 13:20 UTC the
archive served but at ~0.2 MiB/s: a full 91 MB object took 442 s via raven and
491 s pinned to the UVic replica, so it was not route-specific. That regime is
invisible to a timeout-based client -- reads never stall long enough to error
-- and simply produces no progress. For reference, the same client sustained
39–187 MiB/s when the archive was healthy.

**Transient truncated transfers.** Retrievals intermittently ended with
`ProtocolError: Connection broken: IncompleteRead(N bytes read, M more
expected)`, several times a day during degraded periods.

## Separate issue: three objects are stored unreadable *(post-run)*

Distinct from availability, and worth checking against your own fixity
records. Of 172,437 CHIMEFRB baseband objects retrieved, three cannot be
opened by anyone. Each downloads **completely and repeatably at exactly the
length the archive records**, so no client-side integrity check can catch
them; each was verified twice, from two independent networks.

| object | recorded size | failure |
|---|---|---|
| `baseband_41615268_614.h5` | 37,748,736 B | HDF5 superblock declares `stored_eof` 68,243,208 — file is short by 30.5 MB |
| `baseband_143875213_706.h5` | 4,194,304 B | same class: truncated against its own superblock |
| `baseband_20230601064341_844.h5` | 160,077,824 B | `bad object header version number`; md5 `4805230ff6bfb626f85bd6016031e940` |

The first two are internally inconsistent: the stored byte count matches the
inventory, but the HDF5 header inside describes a larger file. That is
consistent with truncation *before or during ingest*, with the truncated
length then recorded as authoritative.

## Impact

Bulk retrieval fails wholesale. Three concurrent archive-processing jobs of ours
died within the same minute when fetches began failing, and cannot restart while
the default locator is down.

*(post-run)* Across the whole campaign the practical cost was roughly a
factor of two in wall-clock: ~60 hours of transfer at healthy rates became
five days. The jobs themselves lost nothing -- they checkpoint and resume --
but each interruption cost the files in flight, and the degraded stretches
produced no progress at all while appearing healthy to every check we had.

## Suggestions

1. Check DNS resolution inside the raven deployment — the dependency it names is
   healthy and externally resolvable, so the failure appears to be raven's own
   resolver or network configuration.
2. Consider returning `503` with `Retry-After` rather than `500` when a
   dependency check fails: a 500 reads as a request fault and is not obviously
   retryable to client libraries.
3. Consider including the failing dependency in the client-facing error.
   `/raven/availability` already knows it, while clients only see
   "unexpected exception calling permissions service(s)".
4. Consider adding `/availability` (not just `/capabilities`) to service
   monitoring, since capabilities stayed 200 for the whole outage.
5. *(post-run)* The later interruptions are invisible even to
   `/raven/availability`, which reported `true` while `cadcget` returned
   nothing. A synthetic end-to-end fetch of a known object would catch both
   those and the degraded-throughput regime, which no status endpoint
   reflected.
6. *(post-run)* Consider a fixity pass over CHIMEFRB baseband: the three
   unreadable objects above are self-consistent by length and would pass any
   size- or checksum-against-record check, so only opening them reveals the
   problem.

## Where to report

Software: **https://github.com/opencadc/storage-inventory** — the OpenCADC
Storage Inventory repository, which contains `raven` (described there as the
global locator service supporting transfer negotiation and direct file GET),
alongside `minoc`, `baldur`, and `luskan`. Issues are enabled.

Note that suggestions 1 is operational (a deployment's DNS), while 2-4 are
software behaviour and belong in the repository. If the archive is still down
when this is filed, CADC operations will act faster than a code tracker; the
issue is still worth filing for the behavioural points.
