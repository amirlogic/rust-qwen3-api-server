//! Inference engine — loads the Qwen3 ONNX model and runs autoregressive generation.
//!
//! Architecture notes
//! ------------------
//! The ONNX export exposes KV cache inputs and outputs so generation can run
//! entirely inside ORT without touching Python.
//!
//! The number of cache layers is discovered dynamically by inspecting the session's input metadata
//! at startup.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context};
use ndarray::{Array0, Array2, ArrayD, Axis, IxDyn};
use ort::{session::Session, value::{Tensor, TensorElementType, ValueType}};
use tokenizers::Tokenizer;
use tracing::{debug, info, instrument};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Public interface
// ─────────────────────────────────────────────────────────────────────────────

/// A loaded, thread-safe inference engine.
#[derive(Clone)]
pub struct Engine(Arc<Mutex<EngineInner>>);

impl Engine {
    /// Load model + tokenizer from `model_dir`.
    ///
    /// Expected layout:
    /// ```text
    /// model_dir/
    ///   tokenizer.json
    ///   onnx/
    ///     model_q4.onnx
    /// ```
    pub fn load(model_dir: &Path, default_max_tokens: usize) -> anyhow::Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let onnx_path = model_dir.join("onnx").join("model_q4.onnx");

        // --- Tokenizer --------------------------------------------------
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {e}", tokenizer_path.display()))?;
        let eos_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .context("Tokenizer does not define the Qwen3 end-of-message token <|im_end|>")?;

        // --- ONNX Runtime -----------------------------------------------
        let env = ort::init()
            .with_name("qwen3")
            .build()
            .context("Failed to initialise ONNX Runtime environment")?;

        let session = Session::builder(&env)
            .context("Failed to create ORT SessionBuilder")?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("Failed to load ONNX model from {}", onnx_path.display()))?;

        // --- Discover cache tensor layout --------------------------------
        let cache_layout = discover_cache_layout(&session)?;
        info!(
            kv_layers = cache_layout.kv_layer_indices.len(),
            conv_layers = cache_layout.conv_layer_indices.len(),
            has_position_ids = cache_layout.has_position_ids,
            "Cache layout discovered"
        );

        Ok(Self(Arc::new(Mutex::new(EngineInner {
            session,
            tokenizer,
            eos_token_id,
            cache_layout,
            default_max_tokens,
        }))))
    }

    /// Generate a response for a chat conversation.
    pub fn generate(
        &self,
        messages: &[crate::server::ChatMessage],
        max_tokens: usize,
    ) -> anyhow::Result<GenerateResult> {
        let mut inner = self.0.lock().map_err(|_| anyhow::anyhow!("Engine mutex poisoned"))?;
        inner.generate(messages, max_tokens)
    }

    pub fn default_max_tokens(&self) -> usize {
        self.0.lock().map(|i| i.default_max_tokens).unwrap_or(512)
    }
}

pub struct GenerateResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Cache layout discovery
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct CacheLayout {
    kv_layer_indices: Vec<usize>,
    conv_layer_indices: Vec<usize>,
    conv_input_prefix: String,
    conv_output_prefix: String,
    /// True when the model expects `position_ids` as an explicit input.
    has_position_ids: bool,
}

fn discover_cache_layout(session: &Session) -> anyhow::Result<CacheLayout> {
    let input_names: Vec<String> = session
        .inputs()
        .iter()
        .map(|outlet| outlet.name().to_string())
        .collect();

    let mut kv_layer_indices: Vec<usize> = input_names
        .iter()
        .filter_map(|name| {
            name.strip_prefix("past_key_values.")
                .and_then(|tail| tail.strip_suffix(".key"))
                .and_then(|index| index.parse::<usize>().ok())
        })
        .collect();
    kv_layer_indices.sort_unstable();

    let conv_input_prefix = if input_names.iter().any(|n| n.starts_with("past_conv.")) {
        "past_conv."
    } else {
        "past_conv_states."
    };
    let mut conv_layer_indices: Vec<usize> = input_names
        .iter()
        .filter_map(|name| {
            name.strip_prefix(conv_input_prefix)
                .and_then(|index| index.parse::<usize>().ok())
        })
        .collect();
    conv_layer_indices.sort_unstable();

    let conv_output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|outlet| outlet.name().to_string())
        .collect();
    let conv_output_prefix = if conv_output_names.iter().any(|n| n.starts_with("present_conv.")) {
        "present_conv."
    } else {
        "present_conv_states."
    };

    let has_position_ids = input_names.iter().any(|n: &String| n == "position_ids");

    Ok(CacheLayout {
        kv_layer_indices,
        conv_layer_indices,
        conv_input_prefix: conv_input_prefix.to_string(),
        conv_output_prefix: conv_output_prefix.to_string(),
        has_position_ids,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Inner engine
// ─────────────────────────────────────────────────────────────────────────────

struct EngineInner {
    session: Session,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    cache_layout: CacheLayout,
    default_max_tokens: usize,
}

impl EngineInner {
    #[instrument(skip(self, messages), fields(max_tokens))]
    fn generate(
        &mut self,
        messages: &[crate::server::ChatMessage],
        max_tokens: usize,
    ) -> anyhow::Result<GenerateResult> {
        // 1. Build prompt text using the Qwen3 chat template
        let prompt = build_chat_prompt(messages);
        debug!(prompt_len = prompt.len(), "Built chat prompt");

        // 2. Tokenise
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenisation failed: {e}"))?;
        let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let prompt_tokens = prompt_ids.len();

        if prompt_tokens == 0 {
            bail!("Empty prompt after tokenisation");
        }

        debug!(prompt_tokens, "Prompt tokenised");

        // 3. Autoregressive generation
        let mut all_ids = prompt_ids.clone();
        let mut generated: Vec<i64> = Vec::new();

        // KV + conv caches — will be populated after the first inference step
        let mut kv_cache: Option<KvCache> = None;

        for step in 0..max_tokens {
            // First step: full prompt; subsequent steps: only the last generated token
            let input_ids_slice: Vec<i64> = if step == 0 {
                all_ids.clone()
            } else {
                vec![*all_ids.last().unwrap()]
            };
            let seq_len = input_ids_slice.len();
            let total_len = all_ids.len();

            // ── Build owned arrays ──────────────────────────────────────────
            let input_ids_arr =
                Array2::from_shape_vec((1, seq_len), input_ids_slice)
                    .context("Failed to build input_ids array")?;

            let attn_mask_arr = Array2::from_elem((1, total_len), 1i64);

            let position_ids_arr = Array2::from_shape_vec(
                (1, seq_len),
                ((total_len - seq_len) as i64..total_len as i64).collect::<Vec<_>>(),
            )
            .context("Failed to build position_ids array")?;

            // ── Assemble named inputs using ort::inputs! macro ─────────────
            // The macro produces a Vec<(Cow<str>, SessionInputValue)>.
            // We start with the fixed inputs then push cache tensors.
            let mut inputs = if self.cache_layout.has_position_ids {
                ort::inputs![
                    "input_ids"      => Tensor::from_array(input_ids_arr)?,
                    "attention_mask" => Tensor::from_array(attn_mask_arr)?,
                    "position_ids"   => Tensor::from_array(position_ids_arr)?,
                ]
            } else {
                ort::inputs![
                    "input_ids"      => Tensor::from_array(input_ids_arr)?,
                    "attention_mask" => Tensor::from_array(attn_mask_arr)?,
                ]
            };

            // This export uses num_logits_to_keep to trim the LM head output.
            // Ask for just the final prompt/token position, which is the only
            // position consumed by the greedy decoding loop below.
            for name in ["num_logits_to_keep", "logits_to_keep"] {
                if self.session.inputs().iter().any(|outlet| outlet.name() == name) {
                    let keep_last_logit = Array0::from_elem((), 1i64);
                    inputs.push((
                        name.into(),
                        Tensor::from_array(keep_last_logit)?.into(),
                    ));
                }
            }

            // ── Cache tensors ───────────────────────────────────────────────
            if let Some(ref cache) = kv_cache {
                for (layer_idx, k, v) in &cache.kv {
                    inputs.push((
                        format!("past_key_values.{layer_idx}.key").into(),
                        Tensor::from_array(k.clone())?.into(),
                    ));
                    inputs.push((
                        format!("past_key_values.{layer_idx}.value").into(),
                        Tensor::from_array(v.clone())?.into(),
                    ));
                }
                for (layer_idx, s) in &cache.conv {
                    inputs.push((
                        format!("{}{layer_idx}", self.cache_layout.conv_input_prefix).into(),
                        Tensor::from_array(s.clone())?.into(),
                    ));
                }
            } else {
                // Step 0 — provide empty zero-element caches for each expected input
                for outlet in self.session.inputs() {
                    let name = outlet.name();
                    if (name.starts_with("past_key_values.")
                        || name.starts_with("past_conv_states.")
                        || name.starts_with("past_conv."))
                        && !inputs.iter().any(|(n, _)| n.as_ref() == name)
                    {
                        // Preserve this cache input's declared head/channel dimensions and
                        // set dynamic dimensions to zero (except a dynamic batch dimension).
                        let empty = empty_cache_tensor(outlet)
                            .with_context(|| format!("Failed to initialize cache input '{name}'"))?;
                        inputs.push((
                            name.to_string().into(),
                            Tensor::from_array(empty)?.into(),
                        ));
                    }
                }
            }

            // ── Run the session ─────────────────────────────────────────────
            let outputs = self
                .session
                .run(inputs)
                .context("ONNX Runtime inference failed")?;

            // ── Extract logits — shape [1, seq_len, vocab_size] ────────────
            let logits_val = outputs
                .get("logits")
                .context("Model did not return 'logits' output")?;

            let logits_view = logits_val
                .try_extract_array::<f32>()
                .context("Failed to extract logits as f32 array")?;

            let last_pos = logits_view.shape()[1] - 1;
            let last_logits = logits_view.index_axis(Axis(1), last_pos);

            // Greedy argmax
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap_or(0);

            debug!(step, next_token, "Generated token");

            all_ids.push(next_token);
            generated.push(next_token);

            // Update KV / conv cache from outputs
            kv_cache = Some(extract_kv_cache(&outputs, &self.cache_layout)?);

            // Stop on EOS
            if next_token as u32 == self.eos_token_id {
                debug!("EOS token reached at step {step}");
                break;
            }
        }

        // 4. Decode generated tokens
        let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        let text = self
            .tokenizer
            .decode(&gen_u32, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {e}"))?;

        Ok(GenerateResult {
            text,
            prompt_tokens,
            completion_tokens: generated.len(),
        })
    }
}

fn empty_cache_tensor(outlet: &ort::value::Outlet) -> anyhow::Result<ArrayD<f32>> {
    let shape = match outlet.dtype() {
        ValueType::Tensor { ty: TensorElementType::Float32, shape, .. } => shape,
        other => bail!("Cache input '{}' must be an f32 tensor, got {other:?}", outlet.name()),
    };

    if shape.is_empty() {
        bail!("Cache input '{}' has no tensor dimensions", outlet.name());
    }

    let dims: Vec<usize> = shape
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            if dim >= 0 {
                dim as usize
            } else if axis == 0 {
                // This server runs one sequence per request.
                1
            } else {
                // Empty past sequence/state; the model fills it on the first step.
                0
            }
        })
        .collect();

    Ok(ArrayD::zeros(IxDyn(&dims)))
}

// ─────────────────────────────────────────────────────────────────────────────
// KV cache helpers
// ─────────────────────────────────────────────────────────────────────────────

struct KvCache {
    kv: Vec<(usize, ArrayD<f32>, ArrayD<f32>)>,
    conv: Vec<(usize, ArrayD<f32>)>,
}

fn extract_kv_cache(
    outputs: &ort::session::SessionOutputs<'_>,
    layout: &CacheLayout,
) -> anyhow::Result<KvCache> {
    let mut kv: Vec<(usize, ArrayD<f32>, ArrayD<f32>)> = Vec::new();
    let mut conv: Vec<(usize, ArrayD<f32>)> = Vec::new();

    for &layer_idx in &layout.kv_layer_indices {
        let key = extract_present_cache(outputs, layer_idx, "key")
            .with_context(|| format!("Missing present key for layer {layer_idx}"))?;
        let val = extract_present_cache(outputs, layer_idx, "value")
            .with_context(|| format!("Missing present value for layer {layer_idx}"))?;
        kv.push((layer_idx, key, val));
    }

    for &layer_idx in &layout.conv_layer_indices {
        let s = extract_f32_array(
            outputs,
            &format!("{}{layer_idx}", layout.conv_output_prefix),
        )
        .with_context(|| format!("Missing present conv state for layer {layer_idx}"))?;
        conv.push((layer_idx, s));
    }

    Ok(KvCache { kv, conv })
}

fn extract_present_cache(
    outputs: &ort::session::SessionOutputs<'_>,
    layer: usize,
    component: &str,
) -> anyhow::Result<ArrayD<f32>> {
    let suffix = format!(".{layer}.{component}");
    let output_name = outputs
        .keys()
        .find(|name| name.starts_with("present") && name.ends_with(&suffix))
        .or_else(|| {
            outputs
                .keys()
                .find(|name| name.starts_with("present") && name.contains(&suffix))
        });

    let Some(output_name) = output_name else {
        let available: Vec<&str> = outputs.keys().collect();
        bail!(
            "No present cache output matched layer {layer} {component}; available outputs: {available:?}"
        );
    };

    extract_f32_array(outputs, output_name)
}

fn extract_f32_array(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> anyhow::Result<ArrayD<f32>> {
    let val = outputs
        .get(name)
        .with_context(|| format!("Output '{name}' not found"))?;
    let view = val
        .try_extract_array::<f32>()
        .with_context(|| format!("Could not extract f32 array for '{name}'"))?;
    Ok(view.to_owned())
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat prompt builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build a prompt string using the Qwen3 chat template.
fn build_chat_prompt(messages: &[crate::server::ChatMessage]) -> String {
    let mut prompt = String::new();
    let mut remaining = messages;

    // The model's template renders a leading system message before the rest.
    if let Some((system, tail)) = messages.split_first().filter(|(msg, _)| msg.role == "system") {
        prompt.push_str(&format!(
            "<|im_start|>system\n<|im_sep|>{}<|im_end|>",
            system.content.trim()
        ));
        remaining = tail;
    }

    for msg in remaining {
        let content = if msg.role == "user" {
            format!("{}\n/no_think", msg.content.trim())
        } else {
            msg.content.trim().to_string()
        };
        prompt.push_str(&format!(
            "<|im_start|>{}\n<|im_sep|>{}<|im_end|>",
            msg.role,
            content
        ));
    }

    prompt.push_str("<|im_start|>assistant\n<|im_sep|>");
    prompt
}



