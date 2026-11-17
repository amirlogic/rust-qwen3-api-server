mod inference;
mod server;

use clap::Parser;
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt};

/// Qwen3-1.7B ONNX API Server — OpenAI-compatible chat completions endpoint
#[derive(Parser, Debug)]
#[command(name = "qwen3-server", version, about)]
struct Args {
    /// Port to listen on
    #[arg(short, long, env = "QWEN3_PORT", default_value = "8080")]
    port: u16,

    /// Host address to bind
    #[arg(long, env = "QWEN3_HOST", default_value = "127.0.0.1")]
    host: String,

    /// Path to the model directory (must contain config.json, tokenizer.json, onnx/model_q4.onnx, etc.)
    #[arg(short, long, env = "QWEN3_MODEL_DIR")]
    model_dir: std::path::PathBuf,

    /// Maximum number of new tokens to generate (default: 512)
    #[arg(long, env = "QWEN3_MAX_TOKENS", default_value = "512")]
    max_tokens: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialise logging. Use RUST_LOG env var to control verbosity.
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    info!(
        host = %args.host,
        port = args.port,
        model_dir = %args.model_dir.display(),
        "Starting Qwen3 API server"
    );

    // Load model and tokenizer (blocking — happens before the server accepts connections)
    let engine = inference::Engine::load(&args.model_dir, args.max_tokens)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {e}"))?;

    info!("Model loaded successfully");

    server::run(args.host, args.port, engine).await
}
