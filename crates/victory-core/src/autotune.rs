//! The `[autotune]` table in `victory.toml`. The GPU crate supplies its
//! hardware defaults and decides whether to read the table at all.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// A startup snapshot of dedicated video memory, not a residency limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VramTier {
    Low,
    Mid,
    High,
    Shared,
    Software,
    Unknown,
}

/// Values not given in the table are left for the startup probe to fill in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutotuneProfile {
    pub ram_budget_mb: Option<u64>,
    pub vram_tier: Option<VramTier>,
    pub hybrid_cpu: Option<bool>,
    pub uma_fast_path: Option<bool>,
}

#[derive(Deserialize, Serialize)]
struct Document {
    autotune: Option<AutotuneProfile>,
}

impl AutotuneProfile {
    /// Read only the opt-in table, independent of the launcher paths.
    pub fn from_toml(text: &str) -> Result<Option<Self>> {
        Ok(toml::from_str::<Document>(text)?.autotune)
    }

    pub fn to_toml(self) -> Result<String> {
        Ok(toml::to_string(&Document { autotune: Some(self) })?)
    }
}
