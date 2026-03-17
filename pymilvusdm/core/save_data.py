import os
import sys
import csv
import json

import h5py
import numpy as np
import yaml

# ScyllaDB imports (optional - only needed if scylla_config is provided)
try:
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.concurrent import execute_concurrent_with_args
    SCYLLA_AVAILABLE = True
except ImportError:
    SCYLLA_AVAILABLE = False


class SaveData:
    def __init__(self, logger, data_dir, timestamp):
        self.logger = logger
        self.data_dir = data_dir
        self.timestamp = timestamp
        self.dirs = os.path.join(self.data_dir, timestamp)

    def save_hdf5_data(self, collection_name, partition_tag, vectors, ids):
        hdf5_filename = os.path.join(self.dirs, collection_name, f"{partition_tag}.h5")
        hdf5_foldername = os.path.dirname(hdf5_filename)
        if not os.path.exists(hdf5_foldername):
            os.makedirs(hdf5_foldername)

        try:
            f = h5py.File(hdf5_filename, "w")
            if type(vectors[0]) == type(b"a"):
                v = []
                for i in vectors:
                    v.append(list(i))
                data = np.array(v, dtype=np.uint8)  # save np.array and dtype=uint8
            else:
                data = np.array(vectors)

            f.create_dataset(name="embeddings", data=data)
            f.create_dataset(name="ids", data=ids)
            self.logger.debug(
                "Successfully saved data of collection: {}/partition: {} data in {}!".format(
                    collection_name, partition_tag, hdf5_filename
                )
            )
            return hdf5_filename
        except Exception as e:
            self.logger.error("Error with {}".format(e))
            sys.exit(1)

    def save_yaml(
        self,
        collection_name,
        partition_tag,
        collection_parameter,
        version,
        save_hdf5_name,
    ):
        try:
            # TODO: `dest_host` and `dest_port` can be set by parameter
            hdf2_yaml = {
                "H2M": {
                    "milvus_version": version,
                    "data_path": [save_hdf5_name],
                    "data_dir": None,
                    "dest_host": "127.0.0.1",
                    "dest_port": 19530,
                    "mode": "skip",
                    "dest_collection_name": collection_name,
                    "dest_partition_name": partition_tag,
                    "collection_parameter": collection_parameter,
                }
            }
            yaml_filename = os.path.join(self.dirs, "yamls", collection_name, f"{partition_tag}.yaml")
            yaml_foldername = os.path.dirname(yaml_filename)
            if not os.path.exists(yaml_foldername):
                os.makedirs(yaml_foldername)

            with open(yaml_filename, "w") as f:
                f.write(yaml.dump(hdf2_yaml))
            self.logger.debug(
                "Successfully saved yamls of collection: {}/partition: {} data in {}!".format(
                    collection_name, partition_tag, yaml_filename
                )
            )
        except Exception as e:
            self.logger.error("Error with {}".format(e))
            sys.exit(1)

    def save_csv_data(self, collection_name, partition_tag, vectors, ids, max_rows=None):
        """Save data to CSV format"""
        csv_filename = os.path.join(self.dirs, collection_name, f"{partition_tag or 'default'}.csv")
        csv_foldername = os.path.dirname(csv_filename)
        if not os.path.exists(csv_foldername):
            os.makedirs(csv_foldername)

        try:
            # Limit rows if specified
            if max_rows and len(vectors) > max_rows:
                vectors = vectors[:max_rows]
                ids = ids[:max_rows]
                self.logger.info(f"Limiting export to {max_rows} rows")

            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header: id, dim_0, dim_1, ..., dim_N
                header = ['id'] + [f'dim_{i}' for i in range(vectors.shape[1])]
                writer.writerow(header)
                
                for id, vec in zip(ids, vectors):
                    row = [int(id)] + vec.tolist()
                    writer.writerow(row)
            
            self.logger.debug(
                "Successfully saved CSV data of collection: {}/partition: {} in {}!".format(
                    collection_name, partition_tag, csv_filename
                )
            )
            return csv_filename
        except Exception as e:
            self.logger.error("Error saving CSV: {}".format(e))
            sys.exit(1)

    def save_json_data(self, collection_name, partition_tag, vectors, ids, max_rows=None):
        """Save data to JSON format"""
        json_filename = os.path.join(self.dirs, collection_name, f"{partition_tag or 'default'}.json")
        json_foldername = os.path.dirname(json_filename)
        if not os.path.exists(json_foldername):
            os.makedirs(json_foldername)

        try:
            # Limit rows if specified
            if max_rows and len(vectors) > max_rows:
                vectors = vectors[:max_rows]
                ids = ids[:max_rows]
                self.logger.info(f"Limiting export to {max_rows} rows")

            data = {
                "collection": collection_name,
                "partition": partition_tag or "_default",
                "dimension": int(vectors.shape[1]),
                "count": len(vectors),
                "data": [
                    {
                        "id": int(id),
                        "vector": vec.tolist()
                    }
                    for id, vec in zip(ids, vectors)
                ]
            }

            with open(json_filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(
                "Successfully saved JSON data of collection: {}/partition: {} in {}!".format(
                    collection_name, partition_tag, json_filename
                )
            )
            return json_filename
        except Exception as e:
            self.logger.error("Error saving JSON: {}".format(e))
            sys.exit(1)

    def save_csv_data_batched(self, collection_name, partition_tag, batch_vectors, batch_ids, is_first_batch=False, max_rows=None, total_written=0):
        """
        Save data to CSV format incrementally (append mode).
        This method writes batches incrementally to avoid loading all data into memory.
        
        Args:
            collection_name: Name of the collection
            partition_tag: Partition tag
            batch_vectors: Numpy array of vectors for this batch
            batch_ids: Numpy array of IDs for this batch
            is_first_batch: True if this is the first batch (write header)
            max_rows: Maximum total rows to write (None for all)
            total_written: Number of rows already written
            
        Returns:
            Total number of rows written after this batch
        """
        csv_filename = os.path.join(self.dirs, collection_name, f"{partition_tag or 'default'}.csv")
        csv_foldername = os.path.dirname(csv_filename)
        if not os.path.exists(csv_foldername):
            os.makedirs(csv_foldername)

        try:
            mode = 'w' if is_first_batch else 'a'
            with open(csv_filename, mode, newline='') as f:
                writer = csv.writer(f)
                
                # Write header only on first batch
                if is_first_batch:
                    header = ['id'] + [f'dim_{i}' for i in range(batch_vectors.shape[1])]
                    writer.writerow(header)
                
                # Check max_rows limit
                if max_rows:
                    remaining = max_rows - total_written
                    if remaining <= 0:
                        return total_written  # Already reached limit
                    if len(batch_vectors) > remaining:
                        batch_vectors = batch_vectors[:remaining]
                        batch_ids = batch_ids[:remaining]
                
                # Write batch rows
                for id, vec in zip(batch_ids, batch_vectors):
                    row = [int(id)] + vec.tolist()
                    writer.writerow(row)
            
            return total_written + len(batch_vectors)
        except Exception as e:
            self.logger.error("Error saving CSV batch: {}".format(e))
            raise

    def save_json_data_batched(self, collection_name, partition_tag, batch_vectors, batch_ids, is_first_batch=False, max_rows=None, total_written=0):
        """
        Save data to JSONL format incrementally (one JSON object per line).
        JSONL is more memory-efficient than standard JSON for large datasets.
        
        Args:
            collection_name: Name of the collection
            partition_tag: Partition tag
            batch_vectors: Numpy array of vectors for this batch
            batch_ids: Numpy array of IDs for this batch
            is_first_batch: True if this is the first batch (write metadata)
            max_rows: Maximum total rows to write (None for all)
            total_written: Number of rows already written
            
        Returns:
            Total number of rows written after this batch
        """
        jsonl_filename = os.path.join(self.dirs, collection_name, f"{partition_tag or 'default'}.jsonl")
        json_foldername = os.path.dirname(jsonl_filename)
        if not os.path.exists(json_foldername):
            os.makedirs(json_foldername)

        try:
            mode = 'w' if is_first_batch else 'a'
            
            # Write metadata file only on first batch
            if is_first_batch:
                metadata_filename = os.path.join(self.dirs, collection_name, f"{partition_tag or 'default'}_metadata.json")
                metadata = {
                    "collection": collection_name,
                    "partition": partition_tag or "_default",
                    "dimension": int(batch_vectors.shape[1]),
                    "format": "jsonl"
                }
                with open(metadata_filename, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            with open(jsonl_filename, mode) as f:
                # Check max_rows limit
                if max_rows:
                    remaining = max_rows - total_written
                    if remaining <= 0:
                        return total_written
                    if len(batch_vectors) > remaining:
                        batch_vectors = batch_vectors[:remaining]
                        batch_ids = batch_ids[:remaining]
                
                # Write each row as a JSON object on a single line
                for id, vec in zip(batch_ids, batch_vectors):
                    row_data = {
                        "id": int(id),
                        "vector": vec.tolist()
                    }
                    f.write(json.dumps(row_data) + '\n')
            
            return total_written + len(batch_vectors)
        except Exception as e:
            self.logger.error("Error saving JSONL batch: {}".format(e))
            raise

    def save_scylla_data_batched(self, collection_name, partition_tag, batch_vectors, batch_ids, 
                                 session, prepared_stmt, concurrency=200, raise_on_error=False):
        """
        Save data to ScyllaDB Vector Search incrementally using async writes.
        
        Args:
            collection_name: Name of the collection (for logging)
            partition_tag: Partition tag (for logging)
            batch_vectors: Numpy array of vectors for this batch
            batch_ids: Numpy array of IDs for this batch
            session: Cassandra/Scylla session object
            prepared_stmt: Prepared INSERT statement
            concurrency: Number of concurrent writes (default 200)
            raise_on_error: Whether to raise on first error (default False)
            
        Returns:
            (total_written, failures): Tuple of count and list of failures
        """
        if not SCYLLA_AVAILABLE:
            raise ImportError("cassandra-driver is not installed. Install it with: pip install cassandra-driver")
        
        try:
            # Convert numpy arrays to list of tuples (id, vector_list)
            # IDs need to be converted to string, vectors to list of floats
            rows = []
            for id_val, vec in zip(batch_ids, batch_vectors):
                # Convert ID to string (adjust if your IDs are already strings)
                id_str = str(int(id_val))
                # Convert vector to list of floats
                vector_list = vec.tolist()
                rows.append((id_str, vector_list))
            
            if not rows:
                return 0, []
            
            # Execute concurrent async writes
            results = execute_concurrent_with_args(
                session, 
                prepared_stmt, 
                rows, 
                concurrency=concurrency, 
                raise_on_first_error=raise_on_error
            )
            
            # Count failures and skipped rows (for INSERT IF NOT EXISTS)
            # results is a list of (success: bool, result_or_exception) tuples
            failures = []
            skipped = 0
            total_written = 0
            
            for i, (ok, res) in enumerate(results):
                if not ok:
                    # Actual error/failure
                    failures.append((i, batch_ids[i], res))
                else:
                    # Check if using INSERT IF NOT EXISTS and row already existed
                    # INSERT IF NOT EXISTS returns a ResultSet with [applied] column
                    # applied=True means row was inserted, applied=False means row already existed
                    try:
                        # For regular INSERT, ResultSet might be empty or None
                        # For INSERT IF NOT EXISTS, ResultSet contains [applied] column
                        if hasattr(res, 'one'):
                            try:
                                row = res.one()
                                if row is not None and hasattr(row, 'applied'):
                                    # INSERT IF NOT EXISTS was used
                                    if row.applied:
                                        total_written += 1
                                    else:
                                        skipped += 1
                                else:
                                    # Regular INSERT (not IF NOT EXISTS) - ResultSet is empty but operation succeeded
                                    total_written += 1
                            except Exception as e:
                                # res.one() might raise if ResultSet is empty (normal for regular INSERT)
                                # This is expected for regular INSERT statements
                                total_written += 1
                        else:
                            # Regular INSERT (not IF NOT EXISTS) - always succeeds if no exception
                            total_written += 1
                    except Exception as e:
                        # If we can't parse the result, log it but assume it succeeded (regular INSERT)
                        self.logger.debug(f"Could not parse result for row {i}: {e}, assuming success")
                        total_written += 1
            
            # Log detailed results
            if failures:
                self.logger.warning(
                    f"ScyllaDB write results for {collection_name}/{partition_tag}: "
                    f"{total_written} written, {skipped} skipped, {len(failures)} failed out of {len(rows)} total"
                )
                failed_ids = [str(int(failed_id)) for _, failed_id, _ in failures]
                self.logger.warning(
                    f"Failed IDs for {collection_name}/{partition_tag}: {','.join(failed_ids)}"
                )
                for idx, failed_id, res in failures[:5]:
                    self.logger.warning(f"Failure at index {idx} for id {int(failed_id)}: {res}")
            else:
                if skipped > 0:
                    self.logger.info(
                        f"ScyllaDB: Inserted {total_written} new rows, skipped {skipped} existing rows "
                        f"for {collection_name}/{partition_tag} (total attempted: {len(rows)})"
                    )
                else:
                    self.logger.info(
                        f"Successfully wrote {total_written} rows to ScyllaDB for "
                        f"{collection_name}/{partition_tag} (total attempted: {len(rows)})"
                    )
            
            return total_written, failures
            
        except Exception as e:
            self.logger.error(f"Error saving batch to ScyllaDB: {e}")
            raise
