# Global raven cannot resolve `ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca` — all `cadcget` downloads fail with HTTP 500

**Observed:** 2026-09-01, from ~11:04 UTC, still ongoing at 21:55 UTC (~11 h)
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

## Impact

Bulk retrieval fails wholesale. Three concurrent archive-processing jobs of ours
died within the same minute when fetches began failing, and cannot restart while
the default locator is down.

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

## Where to report

Software: **https://github.com/opencadc/storage-inventory** — the OpenCADC
Storage Inventory repository, which contains `raven` (described there as the
global locator service supporting transfer negotiation and direct file GET),
alongside `minoc`, `baldur`, and `luskan`. Issues are enabled.

Note that suggestions 1 is operational (a deployment's DNS), while 2-4 are
software behaviour and belong in the repository. If the archive is still down
when this is filed, CADC operations will act faster than a code tracker; the
issue is still worth filing for the behavioural points.
