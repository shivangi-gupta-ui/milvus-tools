import sys
from pymilvusdm.core.read_milvus_data import ReadMilvusDB
from pymilvusdm.core.read_milvus_meta import ReadMilvusMeta
from pymilvusdm.core.save_data import SaveData
from pymilvusdm.core.write_logs import write_log
from pymilvusdm.setting import H2M_YAML
from tqdm import tqdm
from datetime import datetime
import os
import uuid


class MilvusToHDF5():
    def __init__(self, logger, milvusdb, milvus_meta, data_save, milvus_dir, data_dir, mysql_p=None):
        self.logger = logger
        self.milvusdb = milvusdb
        self.milvus_meta = milvus_meta
        self.data_save = data_save
        self.milvus_dir = milvus_dir
        self.data_dir = data_dir
        self.mysql_p = mysql_p

    def get_collection_data(self, collection_name, partition_tags, collection_parameter, version, export_format='hdf5', max_rows=None, batch_size=None):
        pbar = tqdm(partition_tags)
        for partition_tag in pbar:
            pbar.set_description(f"Processing {collection_name}/{partition_tag or 'default'}")
            
            # Use batched processing for CSV/JSON if batch_size is specified
            use_batched = (batch_size is not None and batch_size > 0 and 
                          export_format.lower() in ['csv', 'json'])
            
            if use_batched:
                # Batched processing for memory efficiency
                total_written = 0
                is_first_batch = True
                batch_count = 0
                
                try:
                    for batch_vectors, batch_ids, batch_count in self.milvusdb.read_milvus_file_batched(
                        self.milvus_meta, collection_name, partition_tag, batch_size
                    ):
                        if batch_vectors is None or len(batch_vectors) == 0:
                            continue
                        
                        # Save in the specified format
                        if export_format.lower() == 'csv':
                            total_written = self.data_save.save_csv_data_batched(
                                collection_name, partition_tag, batch_vectors, batch_ids,
                                is_first_batch, max_rows, total_written
                            )
                        elif export_format.lower() == 'json':
                            total_written = self.data_save.save_json_data_batched(
                                collection_name, partition_tag, batch_vectors, batch_ids,
                                is_first_batch, max_rows, total_written
                            )
                        
                        is_first_batch = False
                        
                        # Check if we've reached max_rows
                        if max_rows and total_written >= max_rows:
                            self.logger.info(f"Reached max_rows limit: {max_rows}")
                            break
                        
                        # Clean up batch from memory
                        del batch_vectors
                        del batch_ids
                    
                    if total_written > 0:
                        self.logger.info(f"Successfully exported {total_written} rows from {collection_name}/{partition_tag or 'default'}")
                    else:
                        self.logger.info('The collection: {}/partition: {} has no data.'.format(collection_name, partition_tag))
                        
                except StopIteration:
                    self.logger.info('The collection: {}/partition: {} has no data.'.format(collection_name, partition_tag))
                except Exception as e:
                    self.logger.error(f"Error processing {collection_name}/{partition_tag}: {e}")
                    raise
            else:
                # Non-batched processing (original method) - for HDF5 or when batch_size not specified
                r_vectors, r_ids, r_rows = self.milvusdb.read_milvus_file(self.milvus_meta, collection_name, partition_tag)
                if r_rows == 0:
                    self.logger.info('The collection: {}/partition: {} has no data.'.format(collection_name, partition_tag))
                elif r_rows == len(r_vectors) == len(r_ids):
                    self.logger.debug(
                        "Saving the collection: {}/partition: {} data, total counts(rows, len(vectors), len(ids)) {}".format(
                            collection_name, partition_tag, [r_rows, len(r_vectors), len(r_ids)]))
                    
                    # Save in the specified format
                    if export_format.lower() == 'csv':
                        saved_file = self.data_save.save_csv_data(collection_name, partition_tag, r_vectors, r_ids, max_rows)
                    elif export_format.lower() == 'json':
                        saved_file = self.data_save.save_json_data(collection_name, partition_tag, r_vectors, r_ids, max_rows)
                    else:  # default to hdf5
                        saved_file = self.data_save.save_hdf5_data(collection_name, partition_tag, r_vectors, r_ids)
                        # Only save YAML for HDF5 format (for re-import)
                        self.data_save.save_yaml(collection_name, partition_tag, collection_parameter, version, saved_file)
                else:
                    self.logger.error(
                        "ERROR: The collection: {}/partition: {} data count is not equal!".format(collection_name,
                                                                                                  partition_tag))

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
            self.logger.info("Successfully copied all data.")
        except Exception as e:
            self.logger.error('Error with: {}'.format(e))
            sys.exit(1)


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
