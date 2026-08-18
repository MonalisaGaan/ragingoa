import statistics
import time
from typing import Dict, List


def percentile(
    values: List[float],
    percentage: float
) -> float:

    if not values:
        return 0.0

    values = sorted(values)

    index = int(
        round(
            (percentage / 100)
            * (len(values) - 1)
        )
    )

    return values[index]


def benchmark(
    rag,
    queries: List[str]
) -> Dict:

    latencies = []

    for query in queries:

        start = time.perf_counter()

        rag.retrieve(
            query,
            top_k=5
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        latencies.append(
            elapsed
        )

    return {

        "queries": len(queries),

        "p50_ms": round(
            percentile(
                latencies,
                50
            ),
            2
        ),

        "p70_ms": round(
            percentile(
                latencies,
                70
            ),
            2
        ),

        "p100_ms": round(
            max(latencies),
            2
        ),

        "average_ms": round(
            statistics.mean(latencies),
            2
        )
    }