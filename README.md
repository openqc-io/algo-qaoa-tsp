# QAOA - Travelling Salesman Problem

> Solve the Travelling Salesman Problem using QAOA with QUBO penalty encoding. N cities are mapped to N^2 binary variables (x_{i,p} = city i at position p). Constraints enforce each city visited once and each position filled once. Practical for 3 cities (9 qubits). Based on Lucas, Frontiers in Physics 2:5 (2014).

## At a glance

| | |
|---|---|
| Slug | `qaoa-tsp` |
| Qubits | 9 |
| Industries | logistics |
| Techniques | qaoa |
| Difficulty | intermediate |
| Computation model | gate |
| Access | `open` |
| License | Apache-2.0 |

## Circuit

_Coming soon — runnable implementation pending._

## How to run

### Python SDK

```python
from openqc import OpenQC

qc = OpenQC(api_key="oqc_...")           # see openqc.io → Settings → API Keys
result = qc.algorithm.run("qaoa-tsp", input_data={})
print(result)
```

`input_data` is required (use `{}` if the algorithm takes no parameters).
`run()` polls until the job completes; pass `wait=False` to return the job id immediately.

### HTTP API

```bash
curl -X POST https://openqc.io/v1/jobs/algorithms/qaoa-tsp/run \
  -H "Authorization: Bearer $OPENQC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input_data": {}, "backend": "auto"}'
```

Returns a job id; poll `GET /v1/jobs/{job_id}` until `status=completed`.

### CLI

A dedicated `openqc algorithm run ...` command is on the roadmap (Phase 9). Until then, use the SDK or HTTP examples above.

## References

- https://doi.org/10.3389/fphy.2014.00005

---
Maintained by the OpenQC team. See [CONTRIBUTING.md](https://github.com/openqc-io/algorithms-index/blob/main/CONTRIBUTING.md) for how to propose changes.
