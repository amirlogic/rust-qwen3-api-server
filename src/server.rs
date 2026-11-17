//! Axum HTTP server — OpenAI-compatible chat completions API.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicU64, Ordering};

use axum::{
    Router,
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;
use tracing::{error, info, instrument};

use crate::inference::Engine;

// ─────────────────────────────────────────────────────────────────────────────
// App state
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    engine: Engine,
    request_counter: Arc<AtomicU64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Request / response types
// ─────────────────────────────────────────────────────────────────────────────

/// A single chat message.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Server entry point
// ─────────────────────────────────────────────────────────────────────────────

pub async fn run(host: String, port: u16, engine: Engine) -> anyhow::Result<()> {
    let state = AppState {
        engine,
        request_counter: Arc::new(AtomicU64::new(0)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{host}:{port}");
    info!("Listening on http://{addr}");
    info!("  POST http://{addr}/v1/chat/completions");
    info!("  GET  http://{addr}/v1/models");
    info!("  GET  http://{addr}/health");

    let listener = tokio::net::TcpListener::bind(&addr).await
        .map_err(|e| anyhow::anyhow!("Failed to bind {addr}: {e}"))?;

    axum::serve(listener, app).await
        .map_err(|e| anyhow::anyhow!("Server error: {e}"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "model": "qwen3-1.7b" }))
}

async fn list_models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": "qwen3-1.7b",
            "object": "model",
            "created": 0,
            "owned_by": "local"
        }]
    }))
}

#[instrument(skip(state, request), fields(max_tokens))]
async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    // Validate model field
    if let Some(ref model) = request.model {
        if model != "qwen3-1.7b" && model != "onnx-community/Qwen3-1.7B-ONNX" {
            return api_error(StatusCode::NOT_FOUND, "This server only provides the 'qwen3-1.7b' model.");
        }
    }

    if request.stream {
        return api_error(
            StatusCode::BAD_REQUEST,
            "Streaming is not supported; set \"stream\": false.",
        );
    }

    if request.messages.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "At least one message is required.");
    }

    for msg in &request.messages {
        if !matches!(msg.role.as_str(), "system" | "user" | "assistant") {
            return api_error(
                StatusCode::BAD_REQUEST,
                "Message roles must be 'system', 'user', or 'assistant'.",
            );
        }
        if msg.content.trim().is_empty() {
            return api_error(StatusCode::BAD_REQUEST, "Message content must not be empty.");
        }
    }

    let max_tokens = request.max_tokens.unwrap_or_else(|| state.engine.default_max_tokens());
    if !(1..=4096).contains(&max_tokens) {
        return api_error(StatusCode::BAD_REQUEST, "max_tokens must be between 1 and 4096.");
    }

    let req_id = state.request_counter.fetch_add(1, Ordering::Relaxed) + 1;
    let id = format!("chatcmpl-{req_id}");
    let created = unix_secs();

    info!(id = %id, max_tokens, "Starting inference");

    // Run inference on a blocking thread so we don't stall the async runtime
    let engine = state.engine.clone();
    let messages = request.messages.clone();
    let inference_started = std::time::Instant::now();
    let result = tokio::task::spawn_blocking(move || engine.generate(&messages, max_tokens)).await;
    info!(
        id = %id,
        elapsed_ms = inference_started.elapsed().as_millis() as u64,
        "Inference task returned; building HTTP response"
    );

    match result {
        Ok(Ok(gen)) => {
            info!(
                id = %id,
                prompt_tokens = gen.prompt_tokens,
                completion_tokens = gen.completion_tokens,
                "Inference complete"
            );
            let payload = json!({
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": "qwen3-1.7b",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": gen.text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": gen.prompt_tokens,
                    "completion_tokens": gen.completion_tokens,
                    "total_tokens": gen.prompt_tokens + gen.completion_tokens
                }
            });
            info!(
                id = %id,
                response_chars = payload["choices"][0]["message"]["content"]
                    .as_str().map(str::len).unwrap_or(0),
                "Returning chat completion JSON"
            );
            Json(payload).into_response()
        }
        Ok(Err(e)) => {
            // Include anyhow's source chain so ONNX Runtime input/shape/type
            // errors are visible in the server log instead of only the outer context.
            error!(error = %format!("{e:#}"), "Inference failed");
            api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string())
        }
        Err(e) => {
            error!(error = %e, "Inference task panicked");
            api_error(StatusCode::INTERNAL_SERVER_ERROR, "Inference task panicked.")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn api_error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": status.as_u16()
            }
        })),
    )
        .into_response()
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

