# Local LLM Access

This repository can reach a locally deployed OpenAI-compatible chat endpoint on this machine.

## Endpoint

The model server runs on the Docker **host**, not inside the `nemo-rl` container.
From inside the container use the host-gateway address; from the host use `0.0.0.0`.

| Context | Base URL |
|---------|----------|
| Host machine | `http://0.0.0.0:8000/v1` |
| Inside container (default) | `http://172.17.0.1:8000/v1` |

- Chat completions URL (container): `http://172.17.0.1:8000/v1/chat/completions`
- Model name: `nvidia/nemotron-3-nano`
- Override: set the `MODEL_ENDPOINT` environment variable to use a different address.

## Smoke Test

Use this request to verify the local model is reachable:

```bash
curl -X 'POST' \
  'http://172.17.0.1:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/nemotron-3-nano",
    "messages": [{"role":"user", "content":"Which number is larger, 9.11 or 9.8?"}],
    "max_tokens": 64
  }'
```

## Repo Integration Guidance

Any model adapter in this repo should:

- send `POST` requests to `http://172.17.0.1:8000/v1/chat/completions` (inside the container)
- use the model id `nvidia/nemotron-3-nano`
- send OpenAI-style `messages`
- treat the server as a local dependency running on port `8000`

Example request body:

```json
{
  "model": "nvidia/nemotron-3-nano",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 64
}
```

## Model Cache Location

The model weights are stored through the local NIM cache mapping:

- host cache variable: `$LOCAL_NIM_CACHE`
- container cache path: `/opt/nim/.cache`
- mapping: `$LOCAL_NIM_CACHE:/opt/nim/.cache`
