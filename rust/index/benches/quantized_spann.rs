//! Benchmark for QuantizedSpannIndexWriter (RaBitQ-quantized SPANN) add throughput.

#![recursion_limit = "256"]

mod datasets;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chroma_blockstore::{
    arrow::provider::ArrowBlockfileProvider, provider::BlockfileProvider, BlockfileWriterOptions,
};
use chroma_cache::{new_cache_for_test, new_non_persistent_cache_for_test};
use chroma_distance::DistanceFunction;
use chroma_index::{
    spann::{quantized_spann::QuantizedSpannIndexWriter, types::QuantizedSpannIds},
    usearch::{USearchIndex, USearchIndexProvider},
};
use chroma_storage::{local::LocalStorage, Storage};
use chroma_types::{CollectionUuid, DataRecord, SpannIndexConfig};
use indicatif::{ProgressBar, ProgressStyle};

use datasets::dbpedia::{DbPedia, DATA_LEN, DIMENSION};
use datasets::{format_count, recall_at_k, Query};

// =============================================================================
// CONFIGURATION
// =============================================================================

const BLOCK_SIZE_BYTES: usize = 3 * 1024 * 1024; // 32MB
const BATCH_SIZE: usize = 100_000;
const NUM_BATCHES: usize = 10;
const NUM_THREADS: usize = 16;
const DISTANCE_FUNCTION: DistanceFunction = DistanceFunction::Euclidean;

// =============================================================================
// SPANN Configuration
// =============================================================================

fn spann_config() -> SpannIndexConfig {
    SpannIndexConfig {
        // Write path parameters
        write_nprobe: Some(64),
        nreplica_count: Some(4),
        write_rng_epsilon: Some(8.0),
        write_rng_factor: Some(1.0),

        // Cluster maintenance
        split_threshold: Some(512),
        merge_threshold: Some(128),
        reassign_neighbor_count: Some(4),

        // Commit-time parameters
        center_drift_threshold: Some(0.125),

        // Search parameters
        search_nprobe: Some(64),
        search_rng_epsilon: Some(8.0),
        search_rng_factor: Some(1.0),

        // HNSW parameters
        ef_construction: Some(128),
        ef_search: Some(64),
        max_neighbors: Some(16),

        // Other
        num_centers_to_merge_to: Some(8),
        num_samples_kmeans: Some(1000),
        initial_lambda: Some(100.0),

        quantize: true,
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 1.0 {
        format!("{:.0}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.2}s", secs)
    } else {
        format!("{:.1}m", secs / 60.0)
    }
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== QuantizedSpannIndexWriter Benchmark ===");
    println!(
        "Config: batch_size={}, num_batches={}, threads={}",
        format_count(BATCH_SIZE),
        NUM_BATCHES,
        NUM_THREADS
    );
    println!(
        "Total vectors to index: {}",
        format_count(BATCH_SIZE * NUM_BATCHES)
    );
    println!();

    // Load dataset
    let dataset = DbPedia::load().await?;
    println!(
        "Dataset: {} vectors, {} dimensions",
        format_count(dataset.data_len()),
        dataset.dimension()
    );

    // Setup temp directory and storage
    let tmp_dir = tempfile::tempdir()?;
    let storage = Storage::Local(LocalStorage::new(tmp_dir.path().to_str().unwrap()));

    let collection_id = CollectionUuid::new();
    let config = spann_config();

    // Load ALL vectors upfront (for raw embedding blockfile and batch processing)
    let total_vectors_to_load = (BATCH_SIZE * NUM_BATCHES).min(DATA_LEN);
    println!(
        "Loading {} vectors for raw embedding blockfile...",
        format_count(total_vectors_to_load)
    );
    let load_all_start = Instant::now();
    let all_vectors = dataset.load_range(0, total_vectors_to_load)?;
    let load_all_time = load_all_start.elapsed();
    println!(
        "Loaded {} vectors in {}",
        format_count(all_vectors.len()),
        format_duration(load_all_time)
    );

    // Create raw embedding blockfile with ALL embeddings
    println!("Writing raw embeddings to blockfile...");
    let write_start = Instant::now();

    let block_cache = new_cache_for_test();
    let sparse_index_cache = new_cache_for_test();
    let arrow_blockfile_provider = ArrowBlockfileProvider::new(
        storage.clone(),
        BLOCK_SIZE_BYTES,
        block_cache,
        sparse_index_cache,
        16,
    );
    let blockfile_provider = BlockfileProvider::ArrowBlockfileProvider(arrow_blockfile_provider);

    let raw_embedding_writer = blockfile_provider
        .write::<u32, &DataRecord<'_>>(
            BlockfileWriterOptions::new("".to_string()).ordered_mutations(),
        )
        .await
        .expect("Failed to create raw embedding writer");

    for (id, embedding) in &all_vectors {
        let record = DataRecord {
            id: "",
            embedding: &embedding,
            metadata: None,
            document: None,
        };
        raw_embedding_writer
            .set("", *id, &record)
            .await
            .expect("Failed to write embedding");
    }

    let raw_flusher = raw_embedding_writer
        .commit::<u32, &DataRecord<'_>>()
        .await
        .expect("Failed to commit raw embeddings");
    let raw_embedding_id = raw_flusher.id();
    raw_flusher
        .flush::<u32, &DataRecord<'_>>()
        .await
        .expect("Failed to flush raw embeddings");

    let write_time = write_start.elapsed();
    println!(
        "Wrote {} raw embeddings in {}",
        format_count(all_vectors.len()),
        format_duration(write_time)
    );
    println!();

    // Run batches
    let mut total_vectors = 0usize;
    let mut file_ids: Option<QuantizedSpannIds> = None;
    let total_start = Instant::now();

    #[cfg(feature = "stats")]
    let mut batch_snapshots: Vec<chroma_index::spann::quantized_spann::StatsSnapshot> = Vec::new();

    for batch_idx in 0..NUM_BATCHES {
        let offset = batch_idx * BATCH_SIZE;
        let limit = BATCH_SIZE.min(DATA_LEN.saturating_sub(offset));

        if limit == 0 {
            println!("Batch {}: No more data available", batch_idx);
            break;
        }

        if offset + limit > all_vectors.len() {
            println!("Batch {}: Not enough vectors loaded", batch_idx);
            break;
        }

        // Setup providers (fresh each batch to avoid cache issues)
        let block_cache = new_cache_for_test();
        let sparse_index_cache = new_cache_for_test();
        let arrow_blockfile_provider = ArrowBlockfileProvider::new(
            storage.clone(),
            BLOCK_SIZE_BYTES,
            block_cache,
            sparse_index_cache,
            16,
        );
        let blockfile_provider =
            BlockfileProvider::ArrowBlockfileProvider(arrow_blockfile_provider);

        let usearch_cache = new_non_persistent_cache_for_test();
        let usearch_provider = USearchIndexProvider::new(storage.clone(), usearch_cache);

        // Create or open index
        let index = if let Some(ids) = &file_ids {
            // Open existing index with raw embedding reader
            let raw_reader = blockfile_provider
                .read(
                    chroma_blockstore::arrow::provider::BlockfileReaderOptions::new(
                        raw_embedding_id,
                        "".to_string(),
                    ),
                )
                .await
                .expect("Failed to open raw embedding reader");

            Arc::new(
                QuantizedSpannIndexWriter::<USearchIndex>::open(
                    collection_id,
                    config.clone(),
                    DIMENSION,
                    DISTANCE_FUNCTION,
                    ids.clone(),
                    None,
                    "".to_string(),
                    Some(raw_reader),
                    &blockfile_provider,
                    &usearch_provider,
                )
                .await
                .expect("Failed to open index"),
            )
        } else {
            // Create new index
            Arc::new(
                QuantizedSpannIndexWriter::<USearchIndex>::create(
                    collection_id,
                    config.clone(),
                    DIMENSION,
                    DISTANCE_FUNCTION,
                    None,
                    "".to_string(),
                    &usearch_provider,
                )
                .await
                .expect("Failed to create index"),
            )
        };

        // Get batch vectors from pre-loaded data
        let batch_vectors = &all_vectors[offset..offset + limit];
        let actual_count = batch_vectors.len();

        // Chunk into partitions for parallel processing
        let chunk_size = (actual_count + NUM_THREADS - 1) / NUM_THREADS;
        let chunks = batch_vectors
            .chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>();

        // Progress bar
        let progress = ProgressBar::new(actual_count as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "[Batch {}/{}] {{wide_bar}} {{pos}}/{{len}} [{{elapsed_precise}}<{{eta_precise}}]",
                    batch_idx + 1,
                    NUM_BATCHES
                ))
                .unwrap(),
        );

        // Spawn parallel tasks
        let index_start = Instant::now();
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let index = Arc::clone(&index);
                let progress = progress.clone();
                tokio::spawn(async move {
                    for (id, vec) in chunk {
                        index.add(id, &vec).await.expect("Failed to add vector");
                        progress.inc(1);
                    }
                })
            })
            .collect();

        // Wait for all tasks
        for handle in handles {
            handle.await?;
        }
        progress.finish_and_clear();
        let index_time = index_start.elapsed();

        // Commit and flush this batch
        let commit_start = Instant::now();
        let index = Arc::try_unwrap(index)
            .ok()
            .expect("Index still has references");

        // Capture stats snapshot before commit consumes the index
        #[cfg(feature = "stats")]
        {
            let cluster_sizes = index.cluster_sizes();
            batch_snapshots.push(index.stats().snapshot(&cluster_sizes));
        }

        let flusher = index
            .commit(&blockfile_provider, &usearch_provider)
            .await
            .expect("Failed to commit");
        file_ids = Some(flusher.flush().await.expect("Failed to flush"));
        let commit_time = commit_start.elapsed();

        total_vectors += actual_count;
        let throughput = actual_count as f64 / index_time.as_secs_f64();

        println!(
            "Batch {}: {} vectors | index {} | commit {} | {:.0} vec/s",
            batch_idx + 1,
            format_count(actual_count),
            format_duration(index_time),
            format_duration(commit_time),
            throughput
        );
    }

    let total_time = total_start.elapsed();
    let overall_throughput = total_vectors as f64 / total_time.as_secs_f64();

    // Print method statistics tables
    #[cfg(feature = "stats")]
    {
        println!(
            "{}",
            chroma_index::spann::quantized_spann::format_batch_tables(&batch_snapshots)
        );
    }

    println!("\n=== Indexing Summary ===");
    println!("Total vectors: {}", format_count(total_vectors));
    println!("Total time: {}", format_duration(total_time));
    println!("Overall throughput: {:.0} vec/s", overall_throughput);

    // === Recall Evaluation ===
    println!("\n=== Recall Evaluation ===");

    // Load ground truth queries
    let queries = dataset.queries(DistanceFunction::Cosine)?;
    let k = 100;

    // Filter queries to only those whose ground truth was computed against vectors we indexed
    let valid_queries: Vec<Query> = queries
        .into_iter()
        .filter(|q| q.max_vector_id <= total_vectors as u64)
        .collect();

    println!(
        "Evaluating {} queries (k={})...",
        format_count(valid_queries.len()),
        k
    );

    if valid_queries.is_empty() {
        println!("No valid queries found for the indexed vector count.");
        println!("\nDone!");
        return Ok(());
    }

    // Setup fresh providers for search
    let block_cache = new_cache_for_test();
    let sparse_index_cache = new_cache_for_test();
    let arrow_blockfile_provider = ArrowBlockfileProvider::new(
        storage.clone(),
        BLOCK_SIZE_BYTES,
        block_cache,
        sparse_index_cache,
        16,
    );
    let blockfile_provider = BlockfileProvider::ArrowBlockfileProvider(arrow_blockfile_provider);

    let usearch_cache = new_non_persistent_cache_for_test();
    let usearch_provider = USearchIndexProvider::new(storage.clone(), usearch_cache);

    // Open final index for search
    let file_ids = file_ids.expect("No file_ids after indexing");
    let raw_reader = blockfile_provider
        .read(
            chroma_blockstore::arrow::provider::BlockfileReaderOptions::new(
                raw_embedding_id,
                "".to_string(),
            ),
        )
        .await
        .expect("Failed to open raw embedding reader");

    let index = Arc::new(
        QuantizedSpannIndexWriter::<USearchIndex>::open(
            collection_id,
            config.clone(),
            DIMENSION,
            DISTANCE_FUNCTION,
            file_ids,
            None,
            "".to_string(),
            Some(raw_reader),
            &blockfile_provider,
            &usearch_provider,
        )
        .await
        .expect("Failed to open index for search"),
    );

    // Run parallel recall evaluation
    let recall_start = Instant::now();
    let total_recall_10 = Arc::new(AtomicUsize::new(0));
    let total_recall_100 = Arc::new(AtomicUsize::new(0));
    let num_evaluated = Arc::new(AtomicUsize::new(0));

    let chunk_size = (valid_queries.len() + NUM_THREADS - 1) / NUM_THREADS;
    let query_chunks: Vec<Vec<Query>> = valid_queries
        .chunks(chunk_size)
        .map(|c| c.to_vec())
        .collect();

    let progress = ProgressBar::new(valid_queries.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[Recall] {wide_bar} {pos}/{len} [{elapsed_precise}<{eta_precise}]")
            .unwrap(),
    );

    let handles: Vec<_> = query_chunks
        .into_iter()
        .map(|chunk| {
            let index = Arc::clone(&index);
            let total_recall_10 = Arc::clone(&total_recall_10);
            let total_recall_100 = Arc::clone(&total_recall_100);
            let num_evaluated = Arc::clone(&num_evaluated);
            let progress = progress.clone();
            tokio::spawn(async move {
                let mut local_recall_10_sum: f64 = 0.0;
                let mut local_recall_100_sum: f64 = 0.0;
                let mut local_count: usize = 0;

                for query in chunk {
                    let results = index.search(k, &query.vector).await.expect("Search failed");
                    local_recall_10_sum += recall_at_k(&results.keys, &query.neighbors, 10);
                    local_recall_100_sum += recall_at_k(&results.keys, &query.neighbors, 100);
                    local_count += 1;
                    progress.inc(1);
                }

                total_recall_10.fetch_add(
                    (local_recall_10_sum * 1_000_000.0) as usize,
                    Ordering::Relaxed,
                );
                total_recall_100.fetch_add(
                    (local_recall_100_sum * 1_000_000.0) as usize,
                    Ordering::Relaxed,
                );
                num_evaluated.fetch_add(local_count, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.await?;
    }
    progress.finish_and_clear();

    let recall_time = recall_start.elapsed();
    let total_recall_10_value = total_recall_10.load(Ordering::Relaxed) as f64 / 1_000_000.0;
    let total_recall_100_value = total_recall_100.load(Ordering::Relaxed) as f64 / 1_000_000.0;
    let num_queries = num_evaluated.load(Ordering::Relaxed);
    let avg_recall_10 = if num_queries > 0 {
        total_recall_10_value / num_queries as f64
    } else {
        0.0
    };
    let avg_recall_100 = if num_queries > 0 {
        total_recall_100_value / num_queries as f64
    } else {
        0.0
    };

    println!("Queries evaluated: {}", format_count(num_queries));
    println!(
        "Recall@10: {:.4} | Recall@100: {:.4}",
        avg_recall_10, avg_recall_100
    );
    println!("Evaluation time: {}", format_duration(recall_time));
    println!(
        "Query throughput: {:.0} qps",
        num_queries as f64 / recall_time.as_secs_f64()
    );

    println!("\nDone!");

    Ok(())
}
