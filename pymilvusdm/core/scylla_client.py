from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import ssl


def make_scylla_session(contact_points, port, username, password, keyspace, 
                        local_dc=None, use_ssl=False, ssl_context=None):
    """
    Create and return ScyllaDB/Cassandra cluster and session.
    
    Args:
        contact_points: List of contact point hostnames/IPs
        port: Port number (default 9042)
        username: Username for authentication
        password: Password for authentication
        keyspace: Keyspace name
        local_dc: Local datacenter name (for multi-DC setups)
        use_ssl: Whether to use SSL
        ssl_context: SSL context if use_ssl is True
        
    Returns:
        (cluster, session): Tuple of cluster and session objects
    """
    auth = PlainTextAuthProvider(username=username, password=password)
    
    cluster_kwargs = {
        'contact_points': contact_points,
        'port': port,
        'auth_provider': auth,
        'protocol_version': 4,
    }
    
    if use_ssl and ssl_context:
        cluster_kwargs['ssl_context'] = ssl_context
    elif use_ssl:
        # Create default SSL context if not provided
        ssl_context = ssl.create_default_context()
        cluster_kwargs['ssl_context'] = ssl_context
    
    if local_dc:
        from cassandra.policies import DCAwareRoundRobinPolicy
        cluster_kwargs['load_balancing_policy'] = DCAwareRoundRobinPolicy(local_dc=local_dc)
    
    cluster = Cluster(**cluster_kwargs)
    session = cluster.connect(keyspace)
    
    return cluster, session


def prepare_scylla_insert(session, keyspace, table, id_column='id', embedding_column='embedding', if_not_exists=False):
    """
    Prepare INSERT statement for ScyllaDB vector table.
    
    Args:
        session: Cassandra/Scylla session
        keyspace: Keyspace name
        table: Table name
        id_column: Name of ID column
        embedding_column: Name of embedding/vector column
        if_not_exists: If True, use INSERT IF NOT EXISTS to avoid overwriting existing rows
        
    Returns:
        Prepared statement
    """
    if if_not_exists:
        query = f"INSERT INTO {keyspace}.{table} ({id_column}, {embedding_column}) VALUES (?, ?) IF NOT EXISTS"
    else:
        query = f"INSERT INTO {keyspace}.{table} ({id_column}, {embedding_column}) VALUES (?, ?)"
    return session.prepare(query)


def prepare_scylla_inserts_for_partitions(session, keyspace, table_with_isrc, table_without_isrc, 
                                          id_column='id', embedding_column='embedding', if_not_exists=False):
    """
    Prepare INSERT statements for both partition tables.
    
    Args:
        session: Cassandra/Scylla session
        keyspace: Keyspace name
        table_with_isrc: Table name for vectors_with_isrc partition
        table_without_isrc: Table name for vectors_without_isrc partition
        id_column: Name of ID column
        embedding_column: Name of embedding/vector column
        if_not_exists: If True, use INSERT IF NOT EXISTS to avoid overwriting existing rows
        
    Returns:
        (prepared_stmt_with_isrc, prepared_stmt_without_isrc): Tuple of prepared statements
    """
    stmt_with_isrc = prepare_scylla_insert(session, keyspace, table_with_isrc, id_column, embedding_column, if_not_exists)
    stmt_without_isrc = prepare_scylla_insert(session, keyspace, table_without_isrc, id_column, embedding_column, if_not_exists)
    return stmt_with_isrc, stmt_without_isrc
