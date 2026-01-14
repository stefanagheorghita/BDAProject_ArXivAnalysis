from elasticsearch import Elasticsearch, helpers
import pyarrow.parquet as pq
import time
import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter('ignore', InsecureRequestWarning)

es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "tOx20ZWJwDITugqKOiuq"),
    verify_certs=False,
    request_timeout=300, 
    max_retries=10,       
    retry_on_timeout=True  
)

es.indices.put_settings(index="arxiv", body={"index": {"refresh_interval": "-1"}})

parquet_file = pq.ParquetFile("arxiv_lemmas_with_id.parquet")

def gen_actions():
    for batch in parquet_file.iter_batches(batch_size=5000, columns=["id", "text"]):
        rows = batch.to_pylist()
        for row in rows:
            if not row['text']: continue
            yield {
                "_index": "arxiv",
                "_id": row["id"],
                "_source": {"id": row["id"], "text": row["text"]}
            }

print("Starting resilient bulk ingestion...")

try:

    for success, info in helpers.streaming_bulk(
        es,
        gen_actions(),
        chunk_size=500,     
        max_retries=5,        
        initial_backoff=2,   
        max_backoff=60,     
        raise_on_error=False 
    ):
        if not success:
            print(f"Doc failed: {info}")
            
except Exception as e:
    print(f"Critical Error: {e}")

finally:
    es.indices.put_settings(index="arxiv", body={"index": {"refresh_interval": "1s"}})
