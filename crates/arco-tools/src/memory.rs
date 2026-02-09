//! Memory instrumentation and tracking for Arco optimization stages.
//!
//! This module provides utilities to capture, track, and analyze memory usage
//! across different stages of the optimization process.

use std::time::Instant;
use sysinfo::System;

/// A snapshot of memory state at a specific point in time.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Resident set size in bytes
    pub rss_bytes: u64,
    /// Timestamp when this snapshot was captured
    pub timestamp: Instant,
    /// Name of the stage (e.g., "declare", "optimize")
    pub stage: String,
}

/// Errors produced by memory instrumentation.
#[derive(Debug, Clone)]
pub enum MemoryError {
    ProcessNotFound { pid: u32 },
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::ProcessNotFound { pid } => {
                write!(f, "failed to locate process {}", pid)
            }
        }
    }
}

impl std::error::Error for MemoryError {}

impl MemorySnapshot {
    /// Capture current memory state for a given stage.
    ///
    /// # Errors
    ///
    /// Returns an error if the current process cannot be located.
    pub fn capture(stage: &str) -> Result<Self, MemoryError> {
        let pid = sysinfo::Pid::from(std::process::id() as usize);

        // Only refresh the specific process we care about, not the entire system
        let mut sys = System::new();
        sys.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(&[pid]),
            true,
            sysinfo::ProcessRefreshKind::nothing().with_memory(),
        );

        let process = sys.process(pid).ok_or(MemoryError::ProcessNotFound {
            pid: std::process::id(),
        })?;

        // sysinfo 0.33+ returns memory in bytes directly
        let rss_bytes = process.memory();

        Ok(MemorySnapshot {
            rss_bytes,
            timestamp: Instant::now(),
            stage: stage.to_string(),
        })
    }

    /// Calculate the difference between this snapshot and another.
    ///
    /// Returns the difference in RSS bytes (positive means growth).
    pub fn diff(&self, other: &Self) -> i64 {
        self.rss_bytes as i64 - other.rss_bytes as i64
    }
}

/// A probe for tracking memory usage across multiple stages.
#[derive(Debug)]
pub struct MemoryProbe {
    snapshots: Vec<MemorySnapshot>,
}

impl MemoryProbe {
    /// Create a new memory probe.
    pub fn new() -> Self {
        MemoryProbe {
            snapshots: Vec::new(),
        }
    }

    /// Record a memory snapshot for a stage.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot could not be captured.
    pub fn record(&mut self, stage: &str) -> Result<(), MemoryError> {
        let snapshot = MemorySnapshot::capture(stage)?;
        self.snapshots.push(snapshot);
        Ok(())
    }

    /// Get all recorded snapshots.
    pub fn snapshots(&self) -> &[MemorySnapshot] {
        &self.snapshots
    }

    /// Get the difference between the last two snapshots.
    pub fn last_diff(&self) -> Option<i64> {
        if self.snapshots.len() < 2 {
            return None;
        }
        let last = &self.snapshots[self.snapshots.len() - 1];
        let prev = &self.snapshots[self.snapshots.len() - 2];
        Some(last.diff(prev))
    }
}

impl Default for MemoryProbe {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::{MemoryProbe, MemorySnapshot};
    use std::time::Instant;

    #[test]
    fn test_memory_snapshot_capture() {
        let snapshot =
            MemorySnapshot::capture("test_stage").unwrap_or_else(|err| panic!("{}", err));
        assert_eq!(snapshot.stage, "test_stage");
        assert!(snapshot.rss_bytes > 0);
    }

    #[test]
    fn test_memory_snapshot_diff() {
        let snapshot1 = MemorySnapshot {
            rss_bytes: 1000,
            timestamp: Instant::now(),
            stage: "stage1".to_string(),
        };

        let snapshot2 = MemorySnapshot {
            rss_bytes: 1500,
            timestamp: Instant::now(),
            stage: "stage2".to_string(),
        };

        let diff = snapshot2.diff(&snapshot1);
        assert_eq!(diff, 500);
    }

    #[test]
    fn test_memory_probe() {
        let mut probe = MemoryProbe::new();
        probe
            .record("stage1")
            .unwrap_or_else(|err| panic!("{}", err));
        probe
            .record("stage2")
            .unwrap_or_else(|err| panic!("{}", err));

        assert_eq!(probe.snapshots().len(), 2);
        assert_eq!(probe.snapshots()[0].stage, "stage1");
        assert_eq!(probe.snapshots()[1].stage, "stage2");

        let diff = probe.last_diff();
        assert!(diff.is_some());
    }
}
