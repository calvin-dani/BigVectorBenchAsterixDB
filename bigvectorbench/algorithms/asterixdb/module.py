""" AsterixDB (VTree index) module for BigVectorBench framework. """

import gzip
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import boto3
import numpy as np
import orjson

from bigvectorbench.algorithms.base.module import BaseANN


def metric_mapping(_metric: str) -> str:
    """
    Mapping metric type to AsterixDB similarity name

    Args:
        _metric (str): metric type

    Returns:
        str: AsterixDB similarity name
    """
    _metric = _metric.lower()
    _metric_type = {
        "euclidean": "euclidean",
        "angular": "cosine",
        "dot": "dot",
    }.get(_metric, None)
    if _metric_type is None:
        raise ValueError(f"[AsterixDB] Not support metric type: {_metric}!!!")
    return _metric_type


class AsterixDBVTree(BaseANN):
    """
    AsterixDB with a VTree vector index, driven over the HTTP query API.

    The cluster runs outside BigVectorBench (e.g. the cloud-mode docker compose), so this module is a client
    only. Rows reach AsterixDB through S3 in one of two ways:
    - BVB_LOAD_DEFINITION set: one LOAD of the JSON files already under that prefix (e.g. the bench bucket's
      input/amazon5f/full/data/), authenticated by the node's instance profile.
    - otherwise: train_vec is staged as gzipped JSON parts and each part is loaded with its own statement,
      because reading many large parts in one statement timed out against s3mock.

    Environment:
        BVB_LOAD_CONTAINER / BVB_LOAD_DEFINITION / BVB_LOAD_INCLUDE: bucket, prefix and file pattern (default
            *.json) of files to LOAD directly
        BVB_LOAD_ENDPOINT: S3 endpoint for that LOAD as seen from the AsterixDB nodes; set only for s3mock, which
            takes access keys instead of the instance profile
        ASTERIXDB_URL: query service base URL (default http://127.0.0.1:19002)
        BVB_S3_ENDPOINT: S3 endpoint as seen from this process (default http://localhost:8001, s3mock)
        BVB_S3_ASTERIX_ENDPOINT: the same S3 as seen from the AsterixDB nodes (default http://iceberg-storage:9090)
        BVB_S3_BUCKET / BVB_S3_PREFIX: where the parts go (default bvb / bvb-load/)
        BVB_DATASET: dataset name in the BVB dataverse (default items); its index is <name>_vec
        BVB_PRELOADED: set to 1 when the dataset and its VTree index were already loaded and built on the
            cluster; nothing is dropped, loaded or indexed, only the row count and the index are checked
    """

    DATAVERSE = "BVB"
    DATASET = "items"
    INDEX = "items_vec"
    PART_ROWS = 250_000

    def __init__(self, metric: str, index_param: dict):
        self._metric = metric
        self._similarity = metric_mapping(metric)
        self._index_param = dict(index_param)
        self._url = os.environ.get("ASTERIXDB_URL", "http://127.0.0.1:19002").rstrip("/") + "/query/service"
        self._s3_endpoint = os.environ.get("BVB_S3_ENDPOINT", "http://localhost:8001")
        self._s3_asterix_endpoint = os.environ.get("BVB_S3_ASTERIX_ENDPOINT", "http://iceberg-storage:9090")
        self._s3_bucket = os.environ.get("BVB_S3_BUCKET", "bvb")
        self._s3_prefix = os.environ.get("BVB_S3_PREFIX", "bvb-load/")
        self._s3_access_key = os.environ.get("BVB_S3_ACCESS_KEY", "x")
        self._s3_secret_key = os.environ.get("BVB_S3_SECRET_KEY", "x")
        self._s3_region = os.environ.get("BVB_S3_REGION", "us-west-2")
        self._load_container = os.environ.get("BVB_LOAD_CONTAINER", "")
        self._load_definition = os.environ.get("BVB_LOAD_DEFINITION", "")
        self._load_include = os.environ.get("BVB_LOAD_INCLUDE", "*.json")
        self._load_endpoint = os.environ.get("BVB_LOAD_ENDPOINT", "")
        self.DATASET = os.environ.get("BVB_DATASET", self.DATASET)
        self.INDEX = f"{self.DATASET}_vec"
        self._preloaded = os.environ.get("BVB_PRELOADED", "") == "1"
        if self._load_definition and not self._load_container:
            raise ValueError("[AsterixDB] BVB_LOAD_DEFINITION needs BVB_LOAD_CONTAINER")
        self._s3 = boto3.client(
            "s3",
            endpoint_url=self._s3_endpoint,
            aws_access_key_id=self._s3_access_key,
            aws_secret_access_key=self._s3_secret_key,
            region_name=self._s3_region,
        )
        self._dimension = None
        self.label_names = []
        self._label_pattern = None
        self._min_probe_fraction = 0.75
        self._k_multiplier = 1
        self.name = self._make_name()
        super().__init__()

    def _make_name(self) -> str:
        build = ", ".join(f"{k}:{v}" for k, v in sorted(self._index_param.items()))
        return (
            f"AsterixDB VTree metric:{self._metric}, {build}{', ' if build else ''}"
            f"min_probe_fraction:{self._min_probe_fraction}, k_multiplier:{self._k_multiplier}"
        )

    def _run(self, statement: str, params: dict | None = None, timeout: int = 86400) -> list:
        """
        Run SQL++ statements over the HTTP API and return the results; raise on any error.
        """
        form = {"statement": statement}
        if params:
            form.update({f"${k}": json.dumps(v) for k, v in params.items()})
        req = urllib.request.Request(self._url, data=urllib.parse.urlencode(form).encode())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = json.loads(e.read())
        if body.get("status") != "success":
            raise RuntimeError(f"[AsterixDB] statement failed: {body.get('errors')}\n{statement[:500]}")
        return body.get("results", [])

    def _record(self, i: int, embedding: np.ndarray, labels: np.ndarray | None) -> dict:
        rec = {"id": i}
        if labels is not None:
            for name, value in zip(self.label_names, labels):
                rec[name] = int(value)
        rec["embedding"] = embedding
        return rec

    def _write_part(self, part: int, start: int, embeddings: np.ndarray, labels: np.ndarray | None) -> str:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=3) as g:
            for j in range(len(embeddings)):
                rec = self._record(start + j, embeddings[j], None if labels is None else labels[j])
                g.write(orjson.dumps(rec, option=orjson.OPT_SERIALIZE_NUMPY))
                g.write(b"\n")
        key = f"{self._s3_prefix}part-{part:05d}.json.gz"
        self._s3.put_object(Bucket=self._s3_bucket, Key=key, Body=buf.getvalue())
        return key

    def _s3_with(self, key: str) -> str:
        definition, include = key.rsplit("/", 1)
        return (
            f'("accessKeyId"="{self._s3_access_key}"), ("secretAccessKey"="{self._s3_secret_key}"), '
            f'("region"="{self._s3_region}"), ("serviceEndpoint"="{self._s3_asterix_endpoint}"), '
            f'("pathStyleAddressing"="true"), ("container"="{self._s3_bucket}"), '
            f'("definition"="{definition}/"), ("format"="json"), ("include"="*{include}")'
        )

    def load_data(
        self,
        embeddings: np.array,
        labels: np.ndarray | None = None,
        label_names: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> None:
        self._dimension = embeddings.shape[1]
        self.label_names = list(label_names) if labels is not None and label_names is not None else []
        if self.label_names:
            if label_types is not None and any(t != "int32" for t in label_types):
                raise ValueError(f"[AsterixDB] only int32 labels are supported, got {label_types}")
            self._label_pattern = re.compile(r"\b(" + "|".join(map(re.escape, self.label_names)) + r")\b")
        if self._preloaded:
            self._check_preloaded(embeddings.shape[0])
            return
        fields = ["id: bigint"]
        fields += [f"{n}: int" for n in self.label_names]
        fields.append("embedding: [double]")
        self._run(
            f"DROP DATAVERSE {self.DATAVERSE} IF EXISTS; CREATE DATAVERSE {self.DATAVERSE}; USE {self.DATAVERSE};\n"
            f"CREATE TYPE ItemType AS {{ {', '.join(fields)} }};\n"
            f"CREATE DATASET {self.DATASET}(ItemType) PRIMARY KEY id;"
        )

        n = embeddings.shape[0]
        if self._load_definition:
            self._load_prepared(n)
            return

        # Prepare the next part while AsterixDB loads the current one.
        starts = list(range(0, n, self.PART_ROWS))
        keys = [None] * len(starts)

        def produce(p: int):
            s = starts[p]
            keys[p] = self._write_part(p, s, embeddings[s:s + self.PART_ROWS],
                                       None if labels is None else labels[s:s + self.PART_ROWS])

        produce(0)
        for p in range(len(starts)):
            nxt = threading.Thread(target=produce, args=(p + 1,)) if p + 1 < len(starts) else None
            if nxt:
                nxt.start()
            t0 = time.time()
            if p == 0:
                self._run(f"USE {self.DATAVERSE}; LOAD DATASET {self.DATASET} USING S3 ({self._s3_with(keys[p])});")
            else:
                self._run(
                    f"USE {self.DATAVERSE}; DROP DATASET Ext IF EXISTS;\n"
                    f"CREATE EXTERNAL DATASET Ext(ItemType) USING S3 ({self._s3_with(keys[p])});\n"
                    f"INSERT INTO {self.DATASET} (SELECT VALUE e FROM Ext e); DROP DATASET Ext;"
                )
            self._s3.delete_object(Bucket=self._s3_bucket, Key=keys[p])
            print(f"[AsterixDB] part {p + 1}/{len(starts)} loaded in {time.time() - t0:.0f}s", flush=True)
            if nxt:
                nxt.join()
        self._analyze(n)

    def _load_prepared(self, n: int) -> None:
        """
        Load files already in S3 with one LOAD, instead of staging train_vec. The files must hold the same rows
        as the HDF5 (id = row number in train_vec), or recall is computed against the wrong neighbours.
        """
        if self._load_endpoint:
            auth = (
                f'("accessKeyId"="{self._s3_access_key}"), ("secretAccessKey"="{self._s3_secret_key}"), '
                f'("serviceEndpoint"="{self._load_endpoint}"), ("pathStyleAddressing"="true"), '
            )
        else:
            auth = '("instanceProfile"="true"), '
        t0 = time.time()
        self._run(
            f"USE {self.DATAVERSE}; LOAD DATASET {self.DATASET} USING S3 (\n"
            f'  {auth}("region"="{self._s3_region}"),\n'
            f'  ("container"="{self._load_container}"), ("definition"="{self._load_definition}"),\n'
            f'  ("format"="json"), ("include"="{self._load_include}"));'
        )
        print(f"[AsterixDB] loaded s3://{self._load_container}/{self._load_definition} in {time.time() - t0:.0f}s",
              flush=True)
        self._analyze(n)

    def _analyze(self, n: int) -> None:
        (count,) = self._run(f"USE {self.DATAVERSE}; SELECT VALUE COUNT(*) FROM {self.DATASET};")
        if count != n:
            raise RuntimeError(f"[AsterixDB] {self.DATASET} holds {count:,} rows but train_vec has {n:,}")
        # VTree sizes its build from the dataset statistics, so ANALYZE must precede CREATE INDEX.
        self._run(f"USE {self.DATAVERSE}; ANALYZE DATASET {self.DATASET};")
        self.num_entities = n

    def _check_preloaded(self, n: int) -> None:
        (count,) = self._run(f"USE {self.DATAVERSE}; SELECT VALUE COUNT(*) FROM {self.DATASET};")
        if count != n:
            raise RuntimeError(f"[AsterixDB] {self.DATASET} holds {count:,} rows but train_vec has {n:,}")
        found = self._run(
            "SELECT VALUE i.IndexStructure FROM Metadata.`Index` i "
            f'WHERE i.DataverseName = "{self.DATAVERSE}" AND i.DatasetName = "{self.DATASET}" '
            f'AND i.IndexName = "{self.INDEX}";'
        )
        if found != ["VTREE"]:
            raise RuntimeError(f"[AsterixDB] expected VTREE index {self.INDEX} on {self.DATASET}, found {found}")
        self.num_entities = n

    def create_index(self) -> None:
        if self._preloaded:
            return
        with_params = {"dimension": self._dimension, "similarity": self._similarity, **self._index_param}
        # Filters on fields outside INCLUDE make the planner skip VTree for a full scan (checked with EXPLAIN on
        # 371db29), so every label the filters use is included.
        include = f" INCLUDE ({', '.join(self.label_names)})" if self.label_names else ""
        self._run(
            f"USE {self.DATAVERSE}; CREATE INDEX {self.INDEX} ON {self.DATASET}(embedding VECTOR){include} "
            f"TYPE VTREE WITH {json.dumps(with_params)};"
        )

    def set_query_arguments(self, min_probe_fraction: float = 0.75, k_multiplier: int = 1):
        """
        Set query arguments for ann_distance; both must be literals in the statement, not parameters
        """
        self._min_probe_fraction = float(min_probe_fraction)
        self._k_multiplier = int(k_multiplier)
        self.name = self._make_name()

    def _where(self, filter_expr: str | None) -> str:
        if not filter_expr:
            return ""
        expr = filter_expr.replace("==", "=")
        if self._label_pattern is not None:
            expr = self._label_pattern.sub(lambda m: "r." + m.group(1), expr)
        return f"WHERE {expr} "

    def query(self, v: np.array, n: int, filter_expr: str | None = None) -> list[int]:
        statement = (
            f"USE {self.DATAVERSE}; SELECT VALUE r.id FROM {self.DATASET} r {self._where(filter_expr)}"
            f'ORDER BY ann_distance(r.embedding, $qvec, "{self._similarity}", '
            f"{self._min_probe_fraction}, {self._k_multiplier}) LIMIT {int(n)};"
        )
        return self._run(statement, {"qvec": v.astype(float).tolist()}, timeout=3600)

    def _write_row(self, verb: str, index: int, embeddings: np.ndarray, labels: np.ndarray | None) -> None:
        rec = self._record(index, embeddings, labels if self.label_names else None)
        row = orjson.dumps(rec, option=orjson.OPT_SERIALIZE_NUMPY).decode()
        self._run(f"USE {self.DATAVERSE}; {verb} INTO {self.DATASET} ([{row}]);")

    def insert(self, embeddings: np.ndarray, labels: np.ndarray | None = None) -> None:
        """
        Single insert data

        Args:
            embeddings (np.ndarray): embeddings
            labels (np.ndarray): labels

        Returns:
            None
        """
        # Ids continue after the loaded rows; INSERT rather than UPSERT so a collision fails instead of
        # silently replacing a loaded row.
        self._write_row("INSERT", self.num_entities, embeddings, labels)
        self.num_entities += 1

    def update(
        self, index: int, embeddings: np.ndarray, labels: np.ndarray | None = None
    ) -> None:
        """
        Single update data

        Args:
            index (int): index to update
            embeddings (np.ndarray): embeddings
            labels (np.ndarray): labels

        Returns:
            None
        """
        self._write_row("UPSERT", int(index), embeddings, labels)

    def delete(
        self,
        index: int,
    ) -> None:
        """
        Single delete data

        Args:
            index (int): index to delete

        Returns:
            None
        """
        self._run(f"USE {self.DATAVERSE}; DELETE FROM {self.DATASET} r WHERE r.id = {int(index)};")

    def __str__(self) -> str:
        return self.name
