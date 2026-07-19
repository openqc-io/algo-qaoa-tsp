"""QAOA Travelling Salesman — N^2 binary variables per Lucas (2014), sec. 7.2.

Encoding: qubit v*N + t <=> "city v is visited at position t" (cyclic tour).
QUBO = A * sum_v (1 - sum_t x_vt)^2            (each city visited once)
     + A * sum_t (1 - sum_v x_vt)^2            (each position holds one city)
     + sum_{u!=v} d_uv * sum_t x_ut * x_v,t+1  (tour length, t+1 mod N)
with A = 2 * max(d) + 1 so constraint violations always cost more than any
routing saving (Lucas: A > B * max d, B = 1 here).

Runs as a VQA job: QUBO -> {Z, ZZ} Ising observable (x = (1 - Z)/2);
vortex-compute's QAOA ansatz evolves under this cost Hamiltonian
(Lightning lane, multi-start — the DEFER-AWF-B path).

Inputs (all optional — `{}` runs the 3-city, 9-qubit demo):
  - distance_matrix: N x N matrix, N <= 5 (default: 3-city demo)
  - num_layers:      QAOA depth p, 1..6     (default: 2)

Sandbox: no imports — pure dict/list logic only.
"""

_DEMO_DISTANCES = [
    [0.0, 1.0, 2.0],
    [1.0, 0.0, 1.5],
    [2.0, 1.5, 0.0],
]

# N cities -> N^2 qubits on a statevector VQA lane, and the QUBO build is
# O(N^3). build()/interpret() run inline in vortex-job's event loop (no
# timeout), so cap N BEFORE scanning/building anything.
_MAX_CITIES = 5     # 25 qubits; demo is 3 cities / 9 qubits


def _fail(message):
    """Abort with a descriptive server-side log line. The sandbox exposes
    no exception classes, so raise via a KeyError carrying the message
    (executor logs it; the API returns its generic failure envelope)."""
    return {}[message]


def _finite(x):
    return isinstance(x, (int, float)) and x == x and -1e300 < x < 1e300


def _clean_matrix(raw, fallback):
    """Accept a square finite-numeric matrix (N >= 2); otherwise use the
    fallback. Oversize valid-looking input is rejected loudly instead of
    silently answering the demo problem."""
    if isinstance(raw, list) and len(raw) >= 2:
        if len(raw) > _MAX_CITIES:
            _fail(f"tsp: {len(raw)} cities needs {len(raw) ** 2} qubits; max "
                  f"supported is {_MAX_CITIES} cities ({_MAX_CITIES ** 2} qubits)")
        n = len(raw)
        ok = all(
            isinstance(row, list) and len(row) == n
            and all(_finite(x) for x in row)
            for row in raw
        )
        if ok:
            return [[float(x) for x in row] for row in raw]
    return [list(row) for row in fallback]


def _problem(input_data):
    """Resolve (dist, N, qubo, const). Qubit index = city*N + position."""
    dist = _clean_matrix(input_data.get("distance_matrix"), _DEMO_DISTANCES)
    n = len(dist)
    # Penalty must dominate any routing saving AND stay positive even for
    # (unusual but accepted) negative distances — scale on |d|, not d.
    dmax = max(max(abs(x) for x in row) for row in dist)
    a_pen = 2.0 * dmax + 1.0

    qubo, const = {}, 0.0

    def _add(p, q, coeff):
        key = (p, q) if p <= q else (q, p)
        qubo[key] = qubo.get(key, 0.0) + coeff

    # (1 - sum x)^2 penalties: rows (cities) and columns (positions)
    for v in range(n):
        const += a_pen
        for t in range(n):
            _add(v * n + t, v * n + t, -a_pen)
            for t2 in range(t + 1, n):
                _add(v * n + t, v * n + t2, 2.0 * a_pen)
    for t in range(n):
        const += a_pen
        for v in range(n):
            _add(v * n + t, v * n + t, -a_pen)
            for v2 in range(v + 1, n):
                _add(v * n + t, v2 * n + t, 2.0 * a_pen)
    # Tour length: d_uv when u at position t and v at position t+1 (cyclic)
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            for t in range(n):
                _add(u * n + t, v * n + ((t + 1) % n), dist[u][v])
    return dist, n, qubo, const


def _qubo_to_ising(qubo, const):
    """Map QUBO (x in {0,1}) to Ising Z terms via x = (1 - Z)/2."""
    h, jj, offset = {}, {}, const
    for (a, b), q in qubo.items():
        if a == b:
            offset += q / 2.0
            h[a] = h.get(a, 0.0) - q / 2.0
        else:
            offset += q / 4.0
            h[a] = h.get(a, 0.0) - q / 4.0
            h[b] = h.get(b, 0.0) - q / 4.0
            jj[(a, b)] = jj.get((a, b), 0.0) + q / 4.0
    terms = [
        {"pauli": "Z", "qubits": [i], "coefficient": h[i]}
        for i in sorted(h) if abs(h[i]) > 1e-12
    ] + [
        {"pauli": "ZZ", "qubits": [a, b], "coefficient": jj[(a, b)]}
        for a, b in sorted(jj) if abs(jj[(a, b)]) > 1e-12
    ]
    return terms, offset


class AlgorithmTemplate:

    def build(self, input_data, ctx):
        _dist, n, qubo, const = _problem(input_data)
        terms, _offset = _qubo_to_ising(qubo, const)
        p = input_data.get("num_layers", 2)
        if not isinstance(p, int) or p < 1 or p > 6:
            p = 2
        params = []
        for layer in range(p):
            params.append(f"gamma_{layer}")
            params.append(f"beta_{layer}")
        backend = ctx.get("backend_id", "auto") if isinstance(ctx, dict) else "auto"
        return {
            "type": "vqa",
            "backend_id": backend if backend != "auto" else "openqc-noiseless",
            "provider": "openqc",
            "vqa_config": {
                "ansatz_type": "qaoa",
                "num_qubits": n * n,
                "parameters": params,
                "initial_params": [0.1] * (2 * p),
                "observable": {"type": "hamiltonian", "terms": terms},
                "optimizer": "ADAM",       # gradient lane -> problem-specific QAOA ansatz
                "num_starts": 3,           # non-convex landscape: keep best of 3 restarts
                "max_iterations": 150,
                "shots_per_iteration": 2048,
            },
        }

    def interpret(self, raw_result, input_data):
        dist, n, qubo, const = _problem(input_data)
        _terms, offset = _qubo_to_ising(qubo, const)

        # Sampled path (counts present): decode the best sampled tour.
        counts = raw_result.get("counts") or {}
        if counts:
            best_state, best_cost, best_cnt = None, None, 0
            for state, cnt in counts.items():
                s = state.replace(" ", "")
                bits = [0] * (n * n)
                for q in range(min(n * n, len(s))):
                    bits[q] = int(s[len(s) - 1 - q])
                cost = const
                for (a, b), coeff in qubo.items():
                    cost += coeff * bits[a] * (bits[b] if b != a else 1)
                if best_cost is None or cost < best_cost or (cost == best_cost and cnt > best_cnt):
                    best_state, best_cost, best_cnt = state, cost, cnt
            s = best_state.replace(" ", "")
            tour, position_violations = [], []
            for t in range(n):
                cities = [
                    v for v in range(n)
                    if v * n + t < len(s) and s[len(s) - 1 - (v * n + t)] == "1"
                ]
                tour.append(cities[0] if len(cities) == 1 else None)
                if len(cities) != 1:
                    position_violations.append(t)
            city_violations = [v for v in range(n) if tour.count(v) != 1]
            valid = len(position_violations) == 0 and len(city_violations) == 0
            out = {
                "tour": tour,
                "valid_tour": valid,
                "position_violations": position_violations,
                "city_violations": city_violations,
                "sampled_qubo_cost": round(best_cost, 6),
                "num_cities": n,
            }
            if valid:
                out["tour_length"] = round(
                    sum(dist[tour[t]][tour[(t + 1) % n]] for t in range(n)), 6
                )
            return out

        # VQA path: only the optimized expectation value is available — no
        # sample to decode, so report the cost honestly and say so.
        vr = raw_result.get("vqa_result") or raw_result
        cost = vr.get("optimal_cost")
        out = {
            "num_cities": n,
            "optimal_cost": cost,
            "total_iterations": vr.get("total_iterations", 0),
            "converged": vr.get("converged"),
            "optimal_params": vr.get("optimal_params"),
            "note": (
                "expected_qubo_cost is the optimized QAOA expectation of the "
                "TSP Hamiltonian (constraint penalties + tour length). The "
                "VQA result carries no measurement sample, so no explicit "
                "tour can be decoded here."
            ),
        }
        if isinstance(cost, (int, float)):
            out["expected_qubo_cost"] = round(cost + offset, 6)
        return out
