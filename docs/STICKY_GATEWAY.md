# Sticky forwarding to independent workers

Run one Strata engine per machine. Each worker keeps its own live and parked
conversation cache. One authenticated OpenAI entrance forwards a conversation
to a stable worker without exporting or importing KV snapshots.

The production deployment has one RTX 5090 32GB and two RTX 4090D 48GB workers,
running Qwen3.8-Flash-Next IQ2_XS, INT8 KV and a 32K context. Existing worker
URLs remain usable. Model weights and inference configuration are unchanged.

## Run

Use the workers' existing private `STRATA_API_KEY` and an ordered worker list:

```json
{
  "host": "127.0.0.1",
  "port": 18090,
  "workers": ["http://worker-a:18080", "http://worker-b:18080", "http://worker-c:18080"]
}
```

```bash
python -m serve.kv_gateway --config kv-pool.json
```

For a LAN deployment, bind its LAN address with the key configured. Include the
workers in `NO_PROXY` if the host has an HTTP proxy. The gateway supports
`/v1/chat/completions` (JSON and SSE), `/v1/models`, `/v1/status` and `/health`.
All routes require the key. Use worker URLs for other Strata APIs.

## Conversation affinity

Send a stable `X-Strata-Session` header for each conversation. The gateway hashes
that value to one worker. Without the header, it hashes the model, tools and
initial messages through the first user message. Appending assistant/user turns
keeps the same worker. If clients trim or rewrite that initial history, use an
explicit session header to preserve affinity. Identical initial histories share
a worker; new session identifiers distribute across the ordered worker list.

Routing does not need a session map or files. Restarting the gateway keeps
affinity as long as worker order and count stay unchanged. Requests with no
usable conversation identity use round-robin forwarding. Each worker processes
one request at a time; the gateway admits at most 64 requests and waits at most
120 seconds for a worker slot.

The response includes `X-Strata-Worker` and `X-Strata-Routing: sticky`. Actual
local prefix reuse is reported by `timings.cache_n`. `/v1/status` reports
`routing: sticky`, `kv_transfer: false`, total requests/errors and request counts
by worker. There are no gateway snapshot files, export calls or import calls.

A worker restart or its bounded cache eviction can cause a cold prefill. Worker
failure returns an error; the gateway does not replay generation on another
machine. Changing the worker list remaps conversations and can lose cache hits.

## Validation

```bash
python -m unittest serve.test_kv_gateway
```

The tests exercise JSON/SSE forwarding, automatic affinity across turns, explicit
sessions, distribution across three workers, stable routing after restart,
authentication and status. A transfer method that raises is installed during
the forwarding test to verify that no KV operation is invoked.

For a bounded real-GPU check, send two turns of a unique conversation, interleave
another session, and verify the same worker is selected and `timings.cache_n`
is positive on the continuation. Repeat with sessions assigned to all workers.
Record the answer, worker, cache hit count, request duration and gateway status.
The earlier cross-host transfer measurements remain in
[CROSS_HOST_KV.md](CROSS_HOST_KV.md) as historical evidence.

```bash
python tools/test_sticky_gateway.py --gateway http://gateway:18090 --out evidence.json
```

Measured 2026-10-03 on the three workers above, with a roughly 950-token initial
prompt and a short numerical reply:

| Worker | Cold request | Continuation after another session | Reused / recomputed tokens |
|---|---:|---:|---:|
| RTX 5090 | 0.595 s | 0.207 s | 956 / 24 |
| First RTX 4090D | 0.690 s | 0.203 s | 956 / 24 |
| Second RTX 4090D | 0.864 s | 0.301 s | 956 / 24 |

All conversations kept their assigned worker and returned `749`, then `700`.
Two ordinary OpenAI requests without a session header also kept their worker:
the continuation reused 954 tokens and returned `757` in 0.118 s. Gateway HTTP
errors remained zero and `kv_transfer` was false. All 150 local Python tests
passed. These are bounded routing/cache checks, not a quality or soak benchmark.
An earlier six-request probe failed its cold-answer assertion; its response body
was not recorded. The complete measurements above come from a subsequent unique
probe with response recording added and the same answer assertions retained.
