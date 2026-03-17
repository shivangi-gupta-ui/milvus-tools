import sys
import os
import uuid
import numpy as np
from pymilvusdm.core.read_milvus_data import ReadMilvusDB
from pymilvusdm.core.read_milvus_meta import ReadMilvusMeta
from pymilvusdm.core.save_data import SaveData
from pymilvusdm.core.write_logs import write_log
from pymilvusdm.setting import H2M_YAML
from tqdm import tqdm
from datetime import datetime

# ScyllaDB imports (optional - only needed if scylla_config is provided)
try:
    from pymilvusdm.core.scylla_client import make_scylla_session, prepare_scylla_inserts_for_partitions
    SCYLLA_AVAILABLE = True
except ImportError:
    SCYLLA_AVAILABLE = False


class MilvusToHDF5():
    def __init__(self, logger, milvusdb, milvus_meta, data_save, milvus_dir, data_dir, mysql_p=None, scylla_config=None):
        self.logger = logger
        self.milvusdb = milvusdb
        self.milvus_meta = milvus_meta
        self.data_save = data_save
        self.milvus_dir = milvus_dir
        self.data_dir = data_dir
        self.mysql_p = mysql_p
        self.scylla_config = scylla_config
        
        # Initialize ScyllaDB connection if config provided
        self.scylla_cluster = None
        self.scylla_session = None
        self.scylla_prepared_with_isrc = None
        self.scylla_prepared_without_isrc = None
        
        if scylla_config:
            if not SCYLLA_AVAILABLE:
                raise ImportError("ScyllaDB support requires cassandra-driver. Install with: pip install cassandra-driver")
            try:
                self.logger.info("Initializing ScyllaDB connection...")
                local_dc = scylla_config.get('local_dc')
                if not local_dc:
                    local_dc = 'GCE_ASIA_SOUTH_1'
                
                self.scylla_cluster, self.scylla_session = make_scylla_session(
                    contact_points=scylla_config.get('contact_points'),
                    port=scylla_config.get('port', 9042),
                    username=scylla_config.get('username'),
                    password=scylla_config.get('password'),
                    keyspace=scylla_config.get('keyspace'),
                    local_dc=local_dc,
                    use_ssl=scylla_config.get('use_ssl', False),
                    ssl_context=scylla_config.get('ssl_context')
                )
                
                skip_existing = scylla_config.get('skip_existing', False)
                self.scylla_prepared_with_isrc, self.scylla_prepared_without_isrc = prepare_scylla_inserts_for_partitions(
                    self.scylla_session,
                    keyspace=scylla_config.get('keyspace'),
                    table_with_isrc=scylla_config.get('table_with_isrc', 'audio_vectors_with_isrc'),
                    table_without_isrc=scylla_config.get('table_without_isrc', 'audio_vectors_without_isrc'),
                    id_column=scylla_config.get('id_column', 'id'),
                    embedding_column=scylla_config.get('embedding_column', 'embedding'),
                    if_not_exists=skip_existing
                )
                
                self.logger.info("ScyllaDB connection established successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize ScyllaDB connection: {e}")
                raise

    def get_collection_data(self, collection_name, partition_tags, collection_parameter, version, export_format='hdf5', max_rows=None, batch_size=None):
        # Track totals across all partitions
        total_rows_read_all = 0
        total_unique_all = 0
        total_duplicates_all = 0
        total_rows_written_all = 0
        
        # Track per-partition stats
        partition_stats = {}
        
        pbar = tqdm(partition_tags)
        for partition_tag in pbar:
            partition_name = partition_tag or 'default'
            pbar.set_description(f"Processing {collection_name}/{partition_name}")
            
            # Use batched processing for CSV/JSON/Scylla if batch_size is specified
            use_batched = (batch_size is not None and batch_size > 0 and 
                          (export_format.lower() in ['csv', 'json', 'scylla'] or self.scylla_config))
            
            if use_batched:
                self._process_partition_batched(
                    collection_name, partition_tag, partition_name,
                    export_format, max_rows, batch_size,
                    partition_stats
                )
            else:
                self._process_partition_non_batched(
                    collection_name, partition_tag, partition_name,
                    collection_parameter, version, export_format, max_rows,
                    partition_stats
                )
            
            # Accumulate totals
            stats = partition_stats.get(partition_name, {'read': 0, 'unique': 0, 'duplicates': 0, 'written': 0, 'failed': 0})
            total_rows_read_all += stats['read']
            total_unique_all += stats.get('unique', 0)
            total_duplicates_all += stats.get('duplicates', 0)
            total_rows_written_all += stats['written']
        
        # Final summary
        self._log_final_summary(collection_name, partition_stats, total_rows_read_all, total_unique_all, total_duplicates_all, total_rows_written_all)

    def _process_partition_batched(self, collection_name, partition_tag, partition_name,
                                    export_format, max_rows, batch_size, partition_stats):
        """Batched processing: Read from Milvus → Deduplicate → Write to ScyllaDB, tracking all counts."""
        total_rows_read = 0
        total_unique_rows = 0
        total_duplicate_rows = 0
        total_rows_written = 0
        total_write_failures = 0
        total_file_written = 0
        is_first_batch = True
        batch_number = 0
        failed_batches = []
        
        # Set to track seen vector IDs for deduplication
        seen_ids = set()
        
        self.logger.info(
            f"Starting to read data from Milvus collection '{collection_name}' partition '{partition_name}' "
            f"in batches of {batch_size}"
        )
        
        try:
            read_generator = self.milvusdb.read_milvus_file_batched(
                self.milvus_meta, collection_name, partition_tag, batch_size
            )
            
            for batch_vectors, batch_ids, batch_count in read_generator:
                try:
                    if batch_vectors is None or len(batch_vectors) == 0:
                        self.logger.warning(
                            f"Received empty batch for {collection_name}/{partition_name} - skipping"
                        )
                        continue
                    
                    batch_number += 1
                    batch_row_count = len(batch_ids)
                    total_rows_read += batch_row_count
                    
                    # Log read progress
                    try:
                        batch_id_list = [int(id_val) for id_val in batch_ids.tolist()]
                        first_id = batch_id_list[0] if batch_id_list else "N/A"
                        last_id = batch_id_list[-1] if batch_id_list else "N/A"
                    except Exception:
                        batch_id_list = []
                        first_id = "N/A"
                        last_id = "N/A"
                    
                    # ---- DEDUPLICATION: filter out already-seen IDs ----
                    unique_indices = []
                    batch_duplicates = 0
                    for i, id_val in enumerate(batch_id_list):
                        if id_val not in seen_ids:
                            seen_ids.add(id_val)
                            unique_indices.append(i)
                        else:
                            batch_duplicates += 1
                    
                    batch_unique_count = len(unique_indices)
                    total_unique_rows += batch_unique_count
                    total_duplicate_rows += batch_duplicates
                    
                    self.logger.info(
                        f"[READ] Batch {batch_number} for {collection_name}/{partition_name}: "
                        f"{batch_row_count} rows read, {batch_unique_count} unique, {batch_duplicates} duplicates skipped "
                        f"(total read: {total_rows_read:,}, total unique: {total_unique_rows:,}, total duplicates: {total_duplicate_rows:,}) "
                        f"(first_id={first_id}, last_id={last_id})"
                    )
                    
                    # If no unique rows in this batch, skip writing
                    if batch_unique_count == 0:
                        self.logger.info(
                            f"  Batch {batch_number} has no unique rows - skipping write"
                        )
                        is_first_batch = False
                        continue
                    
                    # Filter vectors and ids to only unique ones
                    unique_indices_arr = np.array(unique_indices)
                    unique_batch_vectors = batch_vectors[unique_indices_arr]
                    unique_batch_ids = batch_ids[unique_indices_arr]
                    
                    # ---- WRITE TO SCYLLADB (only unique rows) ----
                    if self.scylla_config and self.scylla_session:
                        try:
                            # Check max_rows limit for ScyllaDB writes
                            if max_rows and total_rows_written >= max_rows:
                                self.logger.info(f"Reached max_rows limit for ScyllaDB writes: {max_rows}")
                                break
                            
                            # Select the correct prepared statement based on partition
                            if partition_tag == "vectors_with_isrc":
                                prepared_stmt = self.scylla_prepared_with_isrc
                            else:
                                prepared_stmt = self.scylla_prepared_without_isrc
                            
                            # If max_rows is set, trim batch to not exceed limit
                            batch_to_write = unique_batch_vectors
                            batch_ids_to_write = unique_batch_ids
                            if max_rows and total_rows_written + len(unique_batch_vectors) > max_rows:
                                remaining = max_rows - total_rows_written
                                batch_to_write = unique_batch_vectors[:remaining]
                                batch_ids_to_write = unique_batch_ids[:remaining]
                            
                            written, failures = self.data_save.save_scylla_data_batched(
                                collection_name, partition_tag, batch_to_write, batch_ids_to_write,
                                self.scylla_session, prepared_stmt,
                                concurrency=self.scylla_config.get('concurrency', 200),
                                raise_on_error=self.scylla_config.get('raise_on_error', False)
                            )
                            total_rows_written += written
                            total_write_failures += len(failures)
                            
                            self.logger.info(
                                f"[WRITE] Batch {batch_number} for {collection_name}/{partition_name}: "
                                f"sent {len(batch_to_write)} unique rows, wrote {written} "
                                f"(total written: {total_rows_written:,}, total failures: {total_write_failures:,})"
                            )
                            
                            if failures:
                                for idx, failed_id, err in failures[:3]:
                                    self.logger.error(
                                        f"  Write failure at index {idx}, id={int(failed_id)}: {err}"
                                    )
                                if len(failures) > 3:
                                    self.logger.error(f"  ... and {len(failures) - 3} more failures")
                            
                            # Check if we hit limit after write
                            if max_rows and total_rows_written >= max_rows:
                                self.logger.info(f"Reached max_rows limit for ScyllaDB writes: {max_rows}")
                                break
                                
                        except Exception as e:
                            self.logger.error(
                                f"ERROR writing batch {batch_number} to ScyllaDB for {collection_name}/{partition_name}: {e}",
                                exc_info=True
                            )
                            failed_batches.append((batch_number, "ScyllaDB write", str(e)))
                    
                    # ---- WRITE TO FILE (if export_format is not 'scylla') ----
                    if export_format.lower() != 'scylla':
                        try:
                            if export_format.lower() == 'csv':
                                total_file_written = self.data_save.save_csv_data_batched(
                                    collection_name, partition_tag, unique_batch_vectors, unique_batch_ids,
                                    is_first_batch, max_rows, total_file_written
                                )
                            elif export_format.lower() == 'json':
                                total_file_written = self.data_save.save_json_data_batched(
                                    collection_name, partition_tag, unique_batch_vectors, unique_batch_ids,
                                    is_first_batch, max_rows, total_file_written
                                )
                        except Exception as e:
                            self.logger.error(
                                f"ERROR writing batch {batch_number} to file for {collection_name}/{partition_name}: {e}",
                                exc_info=True
                            )
                            failed_batches.append((batch_number, "File write", str(e)))
                    
                    is_first_batch = False
                    
                    # Check max_rows for reads
                    if max_rows and total_rows_read >= max_rows:
                        self.logger.info(f"Reached max_rows limit for reads: {max_rows}")
                        break
                    
                    # Periodic progress logging (every 100k rows read)
                    if total_rows_read % 100000 == 0 and total_rows_read > 0:
                        self.logger.info(
                            f"[PROGRESS] {collection_name}/{partition_name}: "
                            f"Read {total_rows_read:,} | Unique {total_unique_rows:,} | "
                            f"Duplicates {total_duplicate_rows:,} | "
                            f"Written to ScyllaDB {total_rows_written:,} | "
                            f"Write failures {total_write_failures:,}"
                        )
                    
                    # Clean up batch from memory
                    del batch_vectors
                    del batch_ids
                    del unique_batch_vectors
                    del unique_batch_ids
                    
                except Exception as e:
                    self.logger.error(
                        f"ERROR processing batch {batch_number} for {collection_name}/{partition_name}: {e}",
                        exc_info=True
                    )
                    failed_batches.append((batch_number, "Batch processing", str(e)))
                    continue
            
            # ---- Partition summary ----
            self.logger.info("")
            self.logger.info(f"--- Partition '{partition_name}' Summary ---")
            self.logger.info(f"  Total rows read from Milvus:  {total_rows_read:,}")
            self.logger.info(f"  Unique rows:                  {total_unique_rows:,}")
            self.logger.info(f"  Duplicate rows skipped:       {total_duplicate_rows:,}")
            self.logger.info(f"  Rows written to ScyllaDB:     {total_rows_written:,}")
            self.logger.info(f"  Write failures:               {total_write_failures:,}")
            self.logger.info(f"  Total batches processed:      {batch_number}")
            
            if failed_batches:
                self.logger.error(
                    f"  FAILED BATCHES: {len(failed_batches)} batch(es) had errors:"
                )
                for batch_num, error_type, error_msg in failed_batches:
                    self.logger.error(f"    Batch {batch_num} ({error_type}): {error_msg}")
            else:
                self.logger.info(f"  All {batch_number} batches processed successfully")
            self.logger.info("")
            
            partition_stats[partition_name] = {
                'read': total_rows_read,
                'unique': total_unique_rows,
                'duplicates': total_duplicate_rows,
                'written': total_rows_written,
                'failed': total_write_failures,
                'failed_batches': len(failed_batches)
            }
            
        except StopIteration:
            partition_stats[partition_name] = {'read': 0, 'unique': 0, 'duplicates': 0, 'written': 0, 'failed': 0, 'failed_batches': 0}
            self.logger.info(f'The collection: {collection_name}/partition: {partition_tag} has no data.')
        except Exception as e:
            self.logger.error(
                f"CRITICAL ERROR reading from Milvus for {collection_name}/{partition_name}: {e}",
                exc_info=True
            )
            partition_stats[partition_name] = {
                'read': total_rows_read,
                'unique': total_unique_rows,
                'duplicates': total_duplicate_rows,
                'written': total_rows_written,
                'failed': total_write_failures,
                'failed_batches': len(failed_batches),
                'error': str(e)
            }
            self.logger.error(f"Failed partition {partition_name}. Continuing with other partitions...")

    def _process_partition_non_batched(self, collection_name, partition_tag, partition_name,
                                        collection_parameter, version, export_format, max_rows,
                                        partition_stats):
        """Non-batched processing (original method) - for HDF5 or when batch_size not specified."""
        self.logger.info(
            f"Starting to read data from Milvus collection '{collection_name}' partition '{partition_name}' "
            f"(non-batched mode)"
        )
        
        try:
            r_vectors, r_ids, r_rows = self.milvusdb.read_milvus_file(self.milvus_meta, collection_name, partition_tag)
            
            if r_rows == 0:
                partition_stats[partition_name] = {'read': 0, 'unique': 0, 'duplicates': 0, 'written': 0, 'failed': 0, 'failed_batches': 0}
                self.logger.info(f'The collection: {collection_name}/partition: {partition_tag} has no data.')
                return
            
            if r_rows != len(r_vectors) or r_rows != len(r_ids):
                self.logger.error(
                    f"ERROR: {collection_name}/{partition_tag} data count mismatch! "
                    f"Expected {r_rows} rows but got {len(r_vectors)} vectors and {len(r_ids)} ids"
                )
                partition_stats[partition_name] = {
                    'read': min(r_rows, len(r_vectors), len(r_ids)),
                    'unique': 0, 'duplicates': 0, 'written': 0, 'failed': 0, 'failed_batches': 0
                }
                return
            
            # Count unique IDs
            id_list = [int(id_val) for id_val in r_ids.tolist()]
            unique_ids = set(id_list)
            unique_count = len(unique_ids)
            duplicate_count = r_rows - unique_count
            
            self.logger.info(
                f"[READ] Read {r_rows:,} rows from Milvus ({collection_name}/{partition_name}) - "
                f"{unique_count:,} unique, {duplicate_count:,} duplicates"
            )
            
            # Save in the specified format
            try:
                if export_format.lower() == 'csv':
                    self.data_save.save_csv_data(collection_name, partition_tag, r_vectors, r_ids, max_rows)
                elif export_format.lower() == 'json':
                    self.data_save.save_json_data(collection_name, partition_tag, r_vectors, r_ids, max_rows)
                else:  # default to hdf5
                    saved_file = self.data_save.save_hdf5_data(collection_name, partition_tag, r_vectors, r_ids)
                    self.data_save.save_yaml(collection_name, partition_tag, collection_parameter, version, saved_file)
            except Exception as e:
                self.logger.error(
                    f"ERROR saving data for {collection_name}/{partition_name}: {e}",
                    exc_info=True
                )
            
            partition_stats[partition_name] = {
                'read': r_rows, 'unique': unique_count, 'duplicates': duplicate_count,
                'written': 0, 'failed': 0, 'failed_batches': 0
            }
                
        except Exception as e:
            self.logger.error(
                f"CRITICAL ERROR reading from Milvus for {collection_name}/{partition_name}: {e}",
                exc_info=True
            )
            partition_stats[partition_name] = {
                'read': 0, 'unique': 0, 'duplicates': 0, 'written': 0, 'failed': 0, 'failed_batches': 0, 'error': str(e)
            }
            self.logger.error(f"Failed partition {partition_name}. Continuing with other partitions...")

    def _log_final_summary(self, collection_name, partition_stats, total_read, total_unique, total_duplicates, total_written):
        """Log the final summary table with read, unique, duplicate, and write counts per partition."""
        self.logger.info("")
        self.logger.info("=" * 110)
        self.logger.info(f"FINAL SUMMARY: Milvus collection '{collection_name}'")
        self.logger.info("=" * 110)
        self.logger.info(
            f"  {'Partition':<30} {'Total Read':>12} {'Unique':>12} {'Duplicates':>12} {'Written':>12} {'Failures':>12} {'Status'}"
        )
        self.logger.info(
            f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}"
        )
        
        for partition_name in sorted(partition_stats.keys()):
            stats = partition_stats[partition_name]
            read_count = stats['read']
            unique_count = stats.get('unique', 0)
            dup_count = stats.get('duplicates', 0)
            written_count = stats['written']
            failed_count = stats['failed']
            has_error = 'error' in stats
            failed_batches = stats.get('failed_batches', 0)
            
            if has_error:
                status = "ERROR"
            elif failed_count > 0 or failed_batches > 0:
                status = "PARTIAL"
            elif read_count == 0:
                status = "EMPTY"
            else:
                status = "OK"
            
            self.logger.info(
                f"  {partition_name:<30} {read_count:>12,} {unique_count:>12,} {dup_count:>12,} "
                f"{written_count:>12,} {failed_count:>12,} {status}"
            )
        
        self.logger.info(
            f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}"
        )
        
        total_failures = sum(s['failed'] for s in partition_stats.values())
        self.logger.info(
            f"  {'TOTAL':<30} {total_read:>12,} {total_unique:>12,} {total_duplicates:>12,} "
            f"{total_written:>12,} {total_failures:>12,}"
        )
        self.logger.info(f"  Partitions processed: {len(partition_stats)}")
        
        # Duplicate info
        if total_duplicates > 0:
            dup_pct = (total_duplicates / total_read) * 100 if total_read > 0 else 0
            self.logger.warning(
                f"  DUPLICATES: Found {total_duplicates:,} duplicate rows ({dup_pct:.1f}% of total reads). "
                f"These were skipped and NOT written to ScyllaDB."
            )
        
        # Warnings
        error_partitions = [name for name, s in partition_stats.items() if 'error' in s]
        if error_partitions:
            self.logger.error(
                f"  ERROR: {len(error_partitions)} partition(s) had critical errors: {error_partitions}"
            )
        
        zero_read_partitions = [name for name, s in partition_stats.items() if s['read'] == 0 and 'error' not in s]
        if zero_read_partitions:
            self.logger.warning(
                f"  WARNING: {len(zero_read_partitions)} partition(s) had 0 rows read: {zero_read_partitions}"
            )
        
        if total_failures > 0:
            self.logger.warning(
                f"  WARNING: {total_failures:,} total write failures across all partitions. "
                f"Check logs above for details."
            )
        
        # Success message
        if total_unique > 0 and total_written > 0:
            pct = (total_written / total_unique) * 100
            self.logger.info(f"  Write success rate: {pct:.1f}% ({total_written:,} written / {total_unique:,} unique)")
        
        self.logger.info("=" * 110)
        self.logger.info("")

    def read_milvus_data(self, collection_name, partition_tags, export_format='hdf5', max_rows=None, batch_size=None):
        try:
            if not self.milvus_meta.has_collection_meta(collection_name):
                raise Exception("The source collection: {} does not exists.".format(collection_name))

            if not partition_tags:
                partition_tags = [None]
                partition_tags_meta = self.milvus_meta.get_all_partition_tag(collection_name)
                partition_tags += partition_tags_meta
            
            batch_info = f" with batch_size={batch_size}" if batch_size else ""
            self.logger.info(
                "Ready to read all data of collection: {}/partitions: {} in format: {}{}".format(
                    collection_name, partition_tags, export_format, batch_info))

            collection_parameter, version = self.milvus_meta.get_collection_info(collection_name)
            self.get_collection_data(collection_name, partition_tags, collection_parameter, version, export_format, max_rows, batch_size)
            self.logger.info("Successfully processed all data (read from Milvus + write to ScyllaDB).")
        except Exception as e:
            self.logger.error('Error with: {}'.format(e))
            sys.exit(1)
        finally:
            # Clean up ScyllaDB connection
            if self.scylla_cluster:
                self.logger.info("Closing ScyllaDB connection...")
                self.scylla_cluster.shutdown()


if __name__ == "__main__":
    # execute only if run as a script
    milvus_dir = '/Users/root/workspace/milvus10_mysql'
    data_dir = '/Users/root/workspace/test/milvusdm_data'
    mysql_p = {'host': '127.0.0.1', 'user': 'root', 'port': 3306, 'password': '123456', 'database': 'milvus'}

    # collection_name = 'binary_example_collection'
    collection_name = 'folat_example_collection'
    partition_tags = None

    logger = write_log()
    milvusdb = ReadMilvusDB(logger, milvus_dir, mysql_p)
    milvus_meta = ReadMilvusMeta(logger, milvus_dir, mysql_p)
    timestamp = str(uuid.uuid1())
    data_save = SaveData(logger, data_dir, timestamp)

    m2f = MilvusToHDF5(logger, milvusdb, milvus_meta, data_save, milvus_dir, data_dir, mysql_p)
    m2f.read_milvus_data(collection_name, partition_tags)
