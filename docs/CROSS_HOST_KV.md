# Cross-host KV reuse

This fork adds real snapshot transfer between independent single-GPU Strata workers.
A receiving worker restores the donor's KV pages, recurrent/GDN and PLE state,
indexer tails, checkpoints and draft KV. It then skips the matching prompt prefix.
The gateway rotates requests between workers; reuse does not depend on routing a
conversation back to its original machine.

Use the same fork revision, exact GGUF quantization, tokenizer, packed dense and
MTP weights on every worker. IQ2_XS and IQ3_XXS cannot exchange snapshots. Runtime
geometry, KV layout and steering mode must also match. This first implementation
supports single-GPU workers and the deployed 32K context. Layer-split snapshot
transfer is rejected.

## Enable each worker

Build this fork's engine and run its `serve/server.py`. Add these engine arguments
in the existing service config:

```
--conversation-cache-mib 4096 --conversation-cache-slots 4
--conversation-cache-min-free-mib 4096 --prompt-cache-root 512
```

Set `STRATA_API_KEY` to a private shared worker key. Derive a common identity once
per deployment from the actual artifacts and the fork commit, rather than using a
model display name:

```bash
export STRATA_KV_MODEL_ID="$(python tools/kv_model_id.py \
  --config strata-service.json --revision "$(git rev-parse HEAD)")"
```

All workers must produce the same identity. The helper hashes every GGUF shard,
the packed dense/tokenizer artifacts and the MTP pack; paths and derived native
expert rearrangements do not change the identity. Recompute it after changing
weights, tokenizer, steering or revision. Preserve the env variables when spawning
the server. Binary endpoints require both an API key and this explicit opt-in.

The worker adds authenticated endpoints:

| Endpoint | Body | Result |
|---|---|---|
| `POST /v1/cache/export` | Empty | Current completed conversation as binary; `X-Strata-KV-Model` and `X-Strata-KV-SHA256` headers |
| `POST /v1/cache/import` | Binary; repeat both headers | Validated snapshot admitted to the bounded CPU cache |
| `POST /v1/cache/clear` | Empty | Clears live and parked prefixes; subsequent requests recompute |

A busy worker returns 409. Import validates the entire image before a subsequent
request writes GPU state. Failed GPU restores terminate the engine instead of
continuing with partial state. A transfer timeout also ends the engine so a late
control reply cannot corrupt its next token stream.

The engine lock is released before sending a completed control response or
streaming a detached export file. The export response reports
`X-Strata-KV-Engine-Ms` (capture and serialization) and `X-Strata-KV-Hash-Ms`.
Import JSON reports `timings.receive_ms` (upload, checksum and temporary-file
write) and `timings.engine_ms` (CPU decoding, validation and cache admission).
GPU restoration happens on the next matching generation, not during import.

The wire format is versioned, little-endian and contains no pointers or GPU
addresses. SHA-256 checks the HTTP payload; an internal checksum detects damaged
files. File size is capped at 1 GiB, token vectors at 32K, checkpoints at 32 and
layer records at 256. Imports and captures also obey the worker's RAM floor and
cache budget. Temporary files are private and removed after the operation.

## Run the shared entrance

Create a private gateway config:

```json
{
  "host": "127.0.0.1",
  "port": 18090,
  "workers": ["http://worker-a:18080", "http://worker-b:18080", "http://worker-c:18080"],
  "cache_mib": 4096,
  "cache_slots": 16
}
```

Run with the same `STRATA_API_KEY`:

```bash
python -m serve.kv_gateway --config kv-pool.json
```

Clients use the gateway's `/v1` base URL and the existing model name and key.
Ordinary OpenAI chat requests automatically identify their cache branch from the
initial messages and tools. A stable `X-Strata-Session` header can explicitly
identify a conversation. Exact token, image and steering matches in the engine
remain the authority; a mistaken session label never authorizes incompatible KV.

The gateway imports a saved snapshot before a turn moves to another worker and
exports its completed state afterwards. The response reports `X-Strata-Worker`
and `X-Strata-KV-Imported`; **actual reuse** is `timings.cache_n` in the model
response. Imported state without a matching prefix is a cache miss.

Storage is bounded by both bytes and slots, with LRU eviction. The gateway admits
at most 64 requests and serializes each worker's transactions. Snapshots are
private ephemeral files, removed on eviction or normal shutdown. `/v1/status`
reports import/export counts, transferred bytes, cache errors and retained size.
The entrance currently implements OpenAI chat completions, models and status;
use worker URLs directly for the other Strata APIs.

## Validation

CPU checks:

```bash
python -m unittest serve.test_cache_transfer serve.test_kv_gateway tools.test_kv_model_id
c++ -std=c++20 -O2 -DNDEBUG -Iinclude src/core/conversation_wire_test.cpp -o wire-test
./wire-test
```

The C++ test also participates in CMake's conversation tests. The HTTP tests cover
authentication, model/checksum rejection, busy handling, actual binary transport,
round-robin migration, SSE, automatic conversation identification and eviction.

Run a bounded real-GPU check against idle workers:

```bash
python tools/test_cross_host_kv.py --source http://worker-a:18080 \
  --target http://worker-b:18080 --out evidence.json
```

This clears **all** target prefixes before import, so a warm result cannot be
explained by the target's earlier cold run. It checks an identical `749` answer,
more than 512 imported prefix tokens reused and fewer newly computed tokens.

Measured 2026-10-03, this fork on Qwen3.8-Flash-Next IQ2_XS, INT8 KV, 32K context,
CUDA 13.0, RTX 5090 32GB and two RTX 4090D 48GB workers:

| Direction | Snapshot bytes | Export to verifier | Import | Reused / recomputed tokens |
|---|---:|---:|---:|---:|
| 5090 → 4090D with 32GB host RAM | 368,794,765 | 0.715 s | 3.598 s | 942 / 7 |
| 4090D with 32GB host RAM → other 4090D | 368,794,777 | 4.055 s | 1.013 s | 943 / 7 |
| Other 4090D → 5090 | 368,794,777 | 1.153 s | 0.529 s | 943 / 7 |

The verifier ran on the 5090 host; export includes download to it, import includes
upload, checksum and CPU decoding. A three-turn conversation through the entrance
rotated across all three machines and returned `749`, `700`, `708`. Turns two and
three imported real state and reused 957 and 989 tokens; the entrance recorded two
imports, three exports and zero cache errors.

Phase measurements after adding endpoint telemetry, with approximately 369 MB
snapshots and the same verifier placement:

| Direction | Capture / serialize | Export SHA-256 | Download to verifier | Target receive / verify / write | CPU import | Export + import |
|---|---:|---:|---:|---:|---:|---:|
| Other 4090D → 5090 | 0.371 s | 0.146 s | 0.628 s | 0.180 s | 0.339 s | 1.667 s |
| 5090 → 4090D with 32GB host RAM | 0.390 s | 0.146 s | 0.184 s | 3.137 s | 0.453 s | 4.313 s |
| 4090D with 32GB host RAM → other 4090D | 0.710 s | 0.195 s | 3.157 s | 0.629 s | 0.375 s | 5.069 s |

The first network download achieved 587.3 MB/s. The second download was local to
the verifier; its network leg is target receive, about 117.6 MB/s including the
receiver's checksum and file writes. These phases are application measurements,
not isolated network latency. Target prefixes were cleared in both checks;
generation reused 943/945 tokens and recomputed seven, returning `749` in
0.081/0.092 s. Target cold generation took 0.598/0.816 s. Therefore the current
snapshot path costs more than recomputation for this roughly 950-token prompt.
The third direction crosses both network legs via the verifier: its slower leg
downloaded at 116.8 MB/s; it also reused 943 tokens, recomputed seven and returned
`749` in 0.075 s after import.

These are short correctness checks, not quality or sustained-load benchmarks.
The recurrent state makes even short snapshots hundreds of MB. Transfer is
currently synchronous and uncompressed: short chats can be faster to recompute.
Snapshots beyond the cap, missing prefixes, busy peers and transfer errors safely
fall back to prefill at the entrance. Worker inference errors still return errors.
Restarting the entrance discards its caches. Direct calls to workers bypass the
entrance's migration transactions and may reduce cache hit rate. There is no
per-token synchronization or shared GPU address space across machines.

Network measurement in the same deployment: the 5090 negotiated 10Gbps; the
other 4090D's RTL8126 negotiated 5Gbps; the 32GB-RAM worker negotiated 1Gbps.
Ten-second single-TCP iperf3 checks measured 4.707Gbps in both directions between
the first two workers, and 0.935/0.941Gbps to/from the third worker, with zero
retransmits in all four tests. A GPU model or NIC's advertised speed does not
establish the actual inter-host path rate.
