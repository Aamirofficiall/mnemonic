use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::config::MnemonicConfig;
use crate::models::ExtractedEntity;
use crate::store::MemoryStore;

// ─── Gemini API types ───────────────────────────────────────────────────────

#[derive(Serialize)]
struct EmbedRequest {
    content: ContentPart,
}

#[derive(Serialize)]
struct BatchEmbedRequest {
    requests: Vec<EmbedRequest>,
}

#[derive(Serialize)]
struct ContentPart {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: EmbeddingValues,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct BatchEmbedResponse {
    embeddings: Vec<EmbeddingValues>,
}

#[derive(Serialize)]
struct GenerateRequest {
    contents: Vec<GenerateContent>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig,
}

#[derive(Serialize)]
struct GenerateContent {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct GenerationConfig {
    temperature: f32,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
}

#[derive(Deserialize)]
struct GenerateResponse {
    candidates: Option<Vec<Candidate>>,
}

#[derive(Deserialize)]
struct Candidate {
    content: CandidateContent,
}

#[derive(Deserialize)]
struct CandidateContent {
    parts: Vec<CandidatePart>,
}

#[derive(Deserialize)]
struct CandidatePart {
    text: String,
}

// ─── Embedder (Gemini gemini-embedding-001) ───────────────────────────────────

pub struct Embedder {
    client: reqwest::Client,
    api_key: String,
    embed_url: String,
    batch_embed_url: String,
    generate_url: String,
    llm_temperature: f32,
    llm_max_tokens: u32,
    embed_batch_size: usize,
}

impl Embedder {
    pub fn new(config: &MnemonicConfig) -> Result<Self> {
        let api_key = config.gemini_key.clone().ok_or_else(|| anyhow!(
            "No Gemini API key found. Set it in ~/.mnemonic/config.toml, \
             GEMINI_API_KEY env var, or run `mnemonic init`.\n\
             Get a free key at https://aistudio.google.com/apikey"
        ))?;

        info!("Gemini embeddings ready ({}, {} dims)", config.embed_model, config.embed_dims);
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            embed_url: config.embed_url(),
            batch_embed_url: config.batch_embed_url(),
            generate_url: config.generate_url(),
            llm_temperature: config.llm_temperature,
            llm_max_tokens: config.llm_max_tokens,
            embed_batch_size: config.embed_batch_size,
        })
    }

    /// Embed a single text. Returns 768-dim vector.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow!("No tokio runtime"))?;

        let text = text.to_string();
        let fut = self.embed_one_async(&text);

        // Use block_in_place for sync context within async runtime
        tokio::task::block_in_place(|| rt.block_on(fut))
    }

    pub async fn embed_one_async(&self, text: &str) -> Result<Vec<f32>> {
        let body = EmbedRequest {
            content: ContentPart {
                parts: vec![Part { text: text.to_string() }],
            },
        };

        let resp = self.client
            .post(format!("{}?key={}", self.embed_url, self.api_key))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Gemini embed API error {}: {}", status, body));
        }

        let result: EmbedResponse = resp.json().await?;
        Ok(result.embedding.values)
    }

    /// Embed multiple texts in batch (up to 100 per request).
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow!("No tokio runtime"))?;

        let texts = texts.to_vec();
        let fut = self.embed_batch_async(&texts);
        tokio::task::block_in_place(|| rt.block_on(fut))
    }

    async fn embed_batch_async(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::new();

        for chunk in texts.chunks(self.embed_batch_size) {
            let requests: Vec<EmbedRequest> = chunk.iter().map(|t| EmbedRequest {
                content: ContentPart {
                    parts: vec![Part { text: t.clone() }],
                },
            }).collect();

            let body = BatchEmbedRequest { requests };

            let resp = self.client
                .post(format!("{}?key={}", self.batch_embed_url, self.api_key))
                .json(&body)
                .send()
                .await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let err_body = resp.text().await.unwrap_or_default();
                return Err(anyhow!("Gemini batch embed API error {}: {}", status, err_body));
            }

            let result: BatchEmbedResponse = resp.json().await?;
            for emb in result.embeddings {
                all_embeddings.push(emb.values);
            }
        }

        Ok(all_embeddings)
    }

    /// Embed all un-embedded items in the store.
    pub fn embed_all(&self, store: &MemoryStore) -> Result<usize> {
        let items = store.load_unembedded();
        if items.is_empty() {
            return Ok(0);
        }

        let texts: Vec<String> = items.iter().map(|i| i.text.clone()).collect();
        let embeddings = self.embed_batch(&texts)?;

        let mut count = 0;
        for (i, emb) in embeddings.into_iter().enumerate() {
            let item = &items[i];
            store.update_embedding(&item.item_type, &item.id, &emb)?;
            count += 1;
        }

        info!("Embedded {} items via Gemini", count);
        Ok(count)
    }

    /// Use Gemini Flash to extract structured facts from a tool call event.
    pub async fn extract_facts(&self, tool_name: &str, input_json: &str, output_json: &str) -> Result<Vec<ExtractedFact>> {
        let prompt = format!(
r#"You are a memory extraction agent. Analyze this tool call and extract any reusable facts, patterns, gotchas, or conventions worth remembering for future sessions.

Tool: {tool_name}
Input: {input_json}
Output: {output_json}

Rules:
- Only extract facts that would be USEFUL in future sessions
- Skip trivial/obvious information (like "ran a command successfully")
- Each fact should be self-contained (understandable without context)
- Focus on: error patterns, timing data, gotchas, conventions, environment info
- Categorize as: gotcha, pattern, preference, convention, environment, failure
- Extract timing patterns (e.g. "cmake builds take ~60s on this machine")
- Extract error patterns (e.g. "OOM when running all tests at once")
- Extract workflow patterns (e.g. "use nohup for long-running commands")

Respond in JSON array format ONLY (no markdown, no explanation):
[{{"content": "the fact", "category": "pattern", "confidence": 0.85}}]

If nothing worth remembering, respond with: []"#);

        self.call_gemini_json::<Vec<ExtractedFact>>(&prompt).await
    }

    /// Extract entity names from a fact's content for the memory graph
    pub async fn extract_entities(&self, content: &str) -> Result<Vec<ExtractedEntity>> {
        let prompt = format!(
r#"Extract all named entities from this memory fact. Entities include file names, class names, function names, concept names, bug IDs, convention names, RFC names.

Fact: {content}

Rules:
- Extract EVERY identifiable entity (files, classes, functions, concepts, bugs)
- For file paths, extract just the filename (e.g. "comp_renderer.cpp" not the full path)
- For class names, extract the class (e.g. "GpuContext", "SceneCompiler")
- For concepts, extract the key phrase (e.g. "parent transform", "GPU cache")
- For bugs, extract the ID (e.g. "Bug 21", "Bug 30")
- Classify entity_type as: file, class, function, concept, bug, convention, rfc
- Classify relation as: found_in, fixes, relates_to, depends_on, uses

Respond in JSON array format ONLY:
[{{"name": "comp_renderer.cpp", "entity_type": "file", "relation": "found_in"}}]

If no entities found, respond with: []"#);

        self.call_gemini_json::<Vec<ExtractedEntity>>(&prompt).await
    }

    /// Use Gemini Flash to infer root cause from an error→fix pair.
    pub async fn infer_root_cause(&self, error_msg: &str, fix_desc: &str) -> Result<String> {
        let prompt = format!(
r#"Given this error and its fix, what was the root cause? Reply in one concise sentence.

Error: {error_msg}
Fix: {fix_desc}

Root cause:"#);

        let body = GenerateRequest {
            contents: vec![GenerateContent {
                parts: vec![Part { text: prompt }],
            }],
            generation_config: GenerationConfig {
                temperature: 0.0,
                max_output_tokens: 200,
            },
        };

        let resp = self.client
            .post(format!("{}?key={}", self.generate_url, self.api_key))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Ok("Unknown".to_string());
        }

        let result: GenerateResponse = resp.json().await?;
        let text = result.candidates
            .and_then(|c| c.into_iter().next())
            .map(|c| c.content.parts.into_iter().map(|p| p.text).collect::<String>())
            .unwrap_or_else(|| "Unknown".to_string());

        Ok(text.trim().to_string())
    }

    /// Reflect: synthesize insights from a collection of memories
    pub async fn reflect(&self, facts: &[String], topic: Option<&str>) -> Result<String> {
        let facts_text = facts.iter().enumerate()
            .map(|(i, f)| format!("{}. {}", i + 1, f))
            .collect::<Vec<_>>()
            .join("\n");

        let topic_clause = topic
            .map(|t| format!(" Focus on: {}", t))
            .unwrap_or_default();

        let prompt = format!(
r#"You are a memory reflection agent. Analyze these {count} memories and synthesize actionable insights.{topic_clause}

Memories:
{facts_text}

Instructions:
- Identify recurring patterns, common pitfalls, and best practices
- Group related insights together
- Be specific and actionable — not generic advice
- Reference specific files, tools, or conventions mentioned in the memories
- Highlight any contradictions or evolving understanding
- Note what worked vs what didn't

Format your response as a structured analysis with clear sections.
Keep it concise but thorough — focus on what's most useful for future work."#,
            count = facts.len(),
        );

        let body = GenerateRequest {
            contents: vec![GenerateContent {
                parts: vec![Part { text: prompt }],
            }],
            generation_config: GenerationConfig {
                temperature: 0.2,
                max_output_tokens: 4096,
            },
        };

        let resp = self.client
            .post(format!("{}?key={}", self.generate_url, self.api_key))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Gemini reflect API error {}: {}", status, err));
        }

        let result: GenerateResponse = resp.json().await?;
        let text = result.candidates
            .and_then(|c| c.into_iter().next())
            .map(|c| c.content.parts.into_iter().map(|p| p.text).collect::<String>())
            .unwrap_or_else(|| "No insights generated.".to_string());

        Ok(text.trim().to_string())
    }

    /// Merge two facts into one consolidated fact using Gemini
    pub async fn merge_facts(&self, fact_a: &str, fact_b: &str) -> Result<String> {
        let prompt = format!(
r#"Merge these two related memory facts into a single, comprehensive fact that captures all information from both. Keep it concise but complete.

Fact 1: {fact_a}
Fact 2: {fact_b}

Rules:
- Preserve all unique information from both facts
- If one supersedes the other (e.g. "FIXED" vs "deferred"), keep the latest state
- Remove redundant phrasing
- Keep the merged fact self-contained and understandable

Respond with ONLY the merged fact text (no quotes, no explanation):"#);

        let body = GenerateRequest {
            contents: vec![GenerateContent {
                parts: vec![Part { text: prompt }],
            }],
            generation_config: GenerationConfig {
                temperature: 0.0,
                max_output_tokens: 1024,
            },
        };

        let resp = self.client
            .post(format!("{}?key={}", self.generate_url, self.api_key))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Gemini merge API error {}: {}", status, err));
        }

        let result: GenerateResponse = resp.json().await?;
        let text = result.candidates
            .and_then(|c| c.into_iter().next())
            .map(|c| c.content.parts.into_iter().map(|p| p.text).collect::<String>())
            .unwrap_or_else(|| format!("{} | {}", fact_a, fact_b));

        Ok(text.trim().to_string())
    }

    // ─── Private helpers ────────────────────────────────────────────────────

    async fn call_gemini_json<T: for<'de> Deserialize<'de>>(&self, prompt: &str) -> Result<T> {
        let body = GenerateRequest {
            contents: vec![GenerateContent {
                parts: vec![Part { text: prompt.to_string() }],
            }],
            generation_config: GenerationConfig {
                temperature: self.llm_temperature,
                max_output_tokens: self.llm_max_tokens,
            },
        };

        let resp = self.client
            .post(format!("{}?key={}", self.generate_url, self.api_key))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Gemini API error {}: {}", status, err));
        }

        let result: GenerateResponse = resp.json().await?;
        let text = result.candidates
            .and_then(|c| c.into_iter().next())
            .map(|c| c.content.parts.into_iter().map(|p| p.text).collect::<String>())
            .unwrap_or_default();

        // Clean up markdown code blocks
        let clean = text.trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        match serde_json::from_str::<T>(clean) {
            Ok(result) => Ok(result),
            Err(e) => {
                warn!("Failed to parse Gemini JSON: {} — raw: {}", e, clean);
                Err(anyhow!("Failed to parse Gemini response: {}", e))
            }
        }
    }
}

// ─── Extracted fact from Gemini ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ExtractedFact {
    pub content: String,
    pub category: String,
    pub confidence: f64,
}
