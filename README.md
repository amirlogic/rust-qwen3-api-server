# Qwen3 API Server

An OpenAI-compatible HTTP API server that runs the **Qwen3-1.7B** model locally using the **ONNX Runtime** via [`pykeio/ort`](https://github.com/pykeio/ort). Built with [Axum](https://github.com/tokio-rs/axum). Command-line interface only (no GUI required).

---

## Model Preparation

Download the ONNX model from Hugging Face before running:

```bash
# Install huggingface-cli
pip install huggingface_hub[cli]

# Download the Q4 ONNX model (~2.15 GB)
huggingface-cli download onnx-community/Qwen3-1.7B-ONNX \
  --include "config.json" "generation_config.json" "tokenizer.json" "tokenizer_config.json" \
              "onnx/model_q4.onnx" \
  --local-dir ./model
```

This creates the expected directory structure:

```text
model/
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  onnx/
    model_q4.onnx
```

---

## Building the Executable

### Production / Release Build (Recommended)
For model inference, use an optimized release build:

```powershell
cargo build --release
```

- **Output executable:**
  - Windows: `target\release\qwen3-server.exe`
  - Linux / macOS: `target/release/qwen3-server`

### Development / Debug Build
For fast compilation during development:

```powershell
cargo build
```

- **Output executable:**
  - Windows: `target\debug\qwen3-server.exe`
  - Linux / macOS: `target/debug/qwen3-server`

> [!NOTE]
> `ort` uses **load-dynamic** and does not bundle or download the ONNX Runtime shared library. Download the official ONNX Runtime package for your platform and architecture from the [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases). The library must be compatible with ONNX Runtime API 31 (ONNX Runtime 1.31 or newer).
> On Windows, set `ORT_DYLIB_PATH` to the full path of `onnxruntime.dll` from the extracted package (typically under `lib`), or place that DLL next to the executable. On Linux and macOS, set it to the corresponding `libonnxruntime.so` or `libonnxruntime.dylib` file.
>
> **Windows (CPU, no CUDA required):** Download the `onnxruntime-win-x64-<version>.zip` package. Do not choose a package with `-gpu-` in its name unless you specifically need GPU support. Extract the ZIP, then set `ORT_DYLIB_PATH` to the extracted `onnxruntime.dll`. For example, if the ZIP is in Downloads:
>
> ```powershell
> Expand-Archive "$HOME\Downloads\onnxruntime-win-x64-1.31.1.zip" `
>   -DestinationPath "$HOME\onnxruntime" -Force
>
> $dll = Get-ChildItem "$HOME\onnxruntime" -Recurse -Filter onnxruntime.dll |
>   Select-Object -First 1 -ExpandProperty FullName
> $env:ORT_DYLIB_PATH = $dll
> ```
>
> Keep this PowerShell window open and launch the server from it so the environment variable is available:
>
> ```powershell
> .\target\release\qwen3-server.exe --model-dir .\model
> ```
>
> Change the ZIP filename to match the version you downloaded. Ensure the DLL is the official ONNX Runtime library and exports `OrtGetApiBase`; unrelated DLLs with the same filename will fail to load.

---

## Running the Server

### Direct Execution

```powershell
# Windows PowerShell
.\target\release\qwen3-server.exe --model-dir ./model --port 8080

# Linux / macOS
./target/release/qwen3-server --model-dir ./model --port 8080
```

### Using Environment Variables

```powershell
# Windows PowerShell
$env:QWEN3_MODEL_DIR = "./model"
$env:QWEN3_PORT = "8080"
.\target\release\qwen3-server.exe

# Bash
export QWEN3_MODEL_DIR="./model"
export QWEN3_PORT=8080
./target/release/qwen3-server
```

### CLI Options

View all available options:
```powershell
.\target\release\qwen3-server.exe --help
```

| Flag | Env Var | Default | Description |
|---|---|---|---|
| `-p`, `--port` | `QWEN3_PORT` | `8080` | Port to listen on |
| `--host` | `QWEN3_HOST` | `127.0.0.1` | Host address to bind |
| `-m`, `--model-dir` | `QWEN3_MODEL_DIR` | *(Required)* | Path to directory containing model and tokenizer files |
| `--max-tokens` | `QWEN3_MAX_TOKENS` | `512` | Default maximum tokens generated per completion |
| `-h`, `--help` | — | — | Print help information |
| `-V`, `--version` | — | — | Print version |

---

## API Endpoints

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "model": "qwen3-1.7b",
  "status": "ok"
}
```

### 2. List Models
```http
GET /v1/models
```
**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "created": 0,
      "id": "qwen3-1.7b",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

### 3. Chat Completions (OpenAI-compatible)
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "qwen3-1.7b",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user",   "content": "What is the capital of France?" }
  ],
  "max_tokens": 128
}
```

**Example with curl:**

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-1.7b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 128
  }'
```

> [!IMPORTANT]
> Streaming responses (`"stream": true`) are not supported. Set `"stream": false` or omit the field.

---

## Logging

Logging level is controlled via the `RUST_LOG` environment variable:

```powershell
# Windows PowerShell
$env:RUST_LOG = "debug"
.\target\release\qwen3-server.exe --model-dir ./model

# Bash
RUST_LOG=debug ./target/release/qwen3-server --model-dir ./model
```




