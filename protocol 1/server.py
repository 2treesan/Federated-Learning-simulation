# server.py  (Protocol 1)
import os, json, collections
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import numpy as np, torch
from torch import nn
from model import SimpleMLP, get_test_loader
from protocol1 import headtail_str, mask_vec_from_seed, seed_from_base
from shamir import reconstruct_secret

# ===== ENV =====
N       = int(os.environ.get("P1_NCLIENTS", "5"))
T       = int(os.environ.get("P1_T", str((N // 2) + 1)))
ROUNDS  = int(os.environ.get("P1_ROUNDS", "5"))

# Dọn registry rank
try:
    if os.path.exists("p0_registry.json"):
        os.remove("p0_registry.json")
except Exception:
    pass

def _flatten(nds):
    return np.concatenate([a.reshape(-1).astype(np.float32, copy=False) for a in nds], 0)

def _set_model_from_nds(model, nds):
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v) for k, v in zip(keys, nds)}
    model.load_state_dict(state, strict=True)

def _eval_server(nds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)
    _set_model_from_nds(model, nds)
    model.eval()
    tl = get_test_loader(test_batch_size=256)
    crit = nn.CrossEntropyLoss()
    tot, corr, cnt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in tl:
            x, y = x.to(device), y.to(device).long()
            out = model(x)
            loss = crit(out, y)
            tot += loss.item() * x.size(0)
            pred = out.argmax(1, keepdim=True)
            corr += pred.eq(y.view_as(pred)).sum().item()
            cnt += x.size(0)
    return float(tot/cnt), float(corr/cnt)

class P1FedAvg(FedAvg):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.json_path = "protocol1.json"
        self.log = {}
        # shares_pool (GLOBAL): owner -> pair_key "a-b" -> {recipient: y}
        self.shares_pool = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.U1_prev = set()       # vòng 1: không mask
        self.roster_all = set()
        if os.path.exists(self.json_path):
            os.remove(self.json_path)

    def configure_fit(self, server_round, parameters, client_manager):
        ins = super().configure_fit(server_round, parameters, client_manager)
        for _, fitins in ins:
            cfg = dict(fitins.config or {})
            cfg["round_id"] = str(server_round)
            cfg["U1_prev"] = json.dumps(sorted(list(self.U1_prev)))
            cfg["num_clients_total"] = str(max(N, len(self.roster_all) or N))
            cfg["roster_all"] = json.dumps(sorted(list(self.roster_all)))
            fitins.config = cfg
        return ins

    def _dump(self, r, entry):
        self.log[f"epoch {r}"] = entry
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, ensure_ascii=False, indent=2, sort_keys=True)

    def aggregate_fit(self, server_round, results, failures):
        print(f"\n[P1-Server] Round {server_round}: recv {len(results)} results")
        survivors = set()
        masked_nds_list, weights = [], []
        unmasked_flat, masked_flat = {}, {}

        # ===== Thu shares + updates =====
        for client_proxy, fit_res in results:
            cid = getattr(client_proxy, "cid", "unknown")
            nds = parameters_to_ndarrays(fit_res.parameters)  # masked
            print(f"  • From {cid}: preview(masked) = {headtail_str(_flatten(nds))}")
            masked_nds_list.append(nds); weights.append(fit_res.num_examples)

            m = fit_res.metrics or {}
            rank = int(m.get("rank", "-1")); survivors.add(rank); self.roster_all.add(rank)

            uf = np.array(json.loads(m.get("unmasked_flat", "[]")), dtype=np.float32)
            mf = np.array(json.loads(m.get("masked_flat", "[]")), dtype=np.float32)
            unmasked_flat[rank] = uf; masked_flat[rank] = mf

            shares_out = json.loads(m.get("shares_out", "{}"))
            for j_str, per_rec in shares_out.items():
                j = int(j_str)
                a, b = (min(rank, j), max(rank, j))
                pair_key = f"{a}-{b}"
                pool = self.shares_pool[rank].setdefault(pair_key, {})
                for rcpt_str, y in per_rec.items():
                    pool[int(rcpt_str)] = int(y)

        U2 = sorted(list(survivors))
        dropouts = sorted(list(self.U1_prev - survivors))
        print(f"  • Survivors(U2)={U2}; U1_prev={sorted(list(self.U1_prev))}; dropouts={dropouts}")

        masked_total = None
        if masked_nds_list:
            stacked = [_flatten(x) for x in masked_nds_list]
            masked_total = np.sum(np.stack(stacked, 0), 0).astype(np.float32)

        unmasked_total = None
        if unmasked_flat:
            arr = [unmasked_flat[r] for r in U2]
            unmasked_total = np.sum(np.stack(arr, 0), 0).astype(np.float32)

        L = masked_total.size if masked_total is not None else (unmasked_total.size if unmasked_total is not None else 0)

        # ===== REOVERY với fallback: owner=v, thiếu thì owner=u =====
        def _reconstruct_base_for_pair(u: int, v: int) -> np.ndarray | None:
            a, b = (min(u, v), max(u, v))
            pair_key = f"{a}-{b}"

            # owner = v (người rơi)
            owner_pool = self.shares_pool.get(v, {})
            rcpts = owner_pool.get(pair_key, {})
            avail_v = [(rcpt + 1, int(y)) for rcpt, y in rcpts.items()]
            if len(avail_v) >= T:
                base_seed = reconstruct_secret(avail_v[:T])
                return mask_vec_from_seed(L, seed_from_base(base_seed, server_round))

            # fallback owner = u (người sống)
            owner_pool = self.shares_pool.get(u, {})
            rcpts = owner_pool.get(pair_key, {})
            avail_u = [(rcpt + 1, int(y)) for rcpt, y in rcpts.items()]
            if len(avail_u) >= T:
                base_seed = reconstruct_secret(avail_u[:T])
                return mask_vec_from_seed(L, seed_from_base(base_seed, server_round))

            print(f"    [WARN] thiếu shares cho pair {pair_key}: have_v={len(avail_v)} have_u={len(avail_u)} (<{T})")
            return None

        recovery_total = np.zeros(L, dtype=np.float32)
        if server_round >= 2 and dropouts:
            for v in dropouts:
                for u in U2:
                    base = _reconstruct_base_for_pair(u, v)
                    if base is None:
                        continue
                    sign = +1.0 if u < v else -1.0
                    recovery_total += sign * base

        masked_minus_recovery = masked_total - recovery_total if masked_total is not None else None
        if masked_minus_recovery is not None and unmasked_total is not None:
            err = float(np.linalg.norm(masked_minus_recovery - unmasked_total))
            print(f"  • ||(masked - recovery) - unmasked||_2 = {err:.6e}")

        # ===== FedAvg sau khi gỡ mask theo từng client =====
        corr_per_client = {u: np.zeros(L, dtype=np.float32) for u in U2}
        if server_round >= 2 and dropouts:
            for v in dropouts:
                for u in U2:
                    base = _reconstruct_base_for_pair(u, v)
                    if base is None:
                        continue
                    sign = +1.0 if u < v else -1.0
                    corr_per_client[u] += sign * base

        corrected_nds_list = []
        for nds, w, u in zip(masked_nds_list, weights, U2):
            corr = corr_per_client[u]
            flat = _flatten(nds) - corr
            parts, i = [], 0
            for arr in nds:
                sz = arr.size
                parts.append(flat[i:i+sz].reshape(arr.shape).astype(np.float32)); i += sz
            corrected_nds_list.append((parts, w))

        totw = float(sum(w for _, w in corrected_nds_list)) or 1.0
        accum = None
        for nds, w in corrected_nds_list:
            if accum is None:
                accum = [a.astype(np.float32) * (w / totw) for a in nds]
            else:
                for i in range(len(nds)):
                    accum[i] += nds[i].astype(np.float32) * (w / totw)

        loss, acc = _eval_server(accum)
        print("  • global preview:", headtail_str(_flatten(accum)))
        print(f"  • Server Eval: loss={loss:.6f}, acc={acc:.4f}")

        entry = {}
        for r in U2:
            entry[f"client {r} (unmasked)"] = headtail_str(unmasked_flat[r])
            entry[f"client {r} (masked)"]   = headtail_str(masked_flat[r])
        entry["U1_prev"] = sorted(list(self.U1_prev))
        entry["dropouts"] = dropouts
        entry["sum (unmasked)"] = headtail_str(unmasked_total)
        entry["sum (masked)"]   = headtail_str(masked_total)
        entry["recovery sum (masked)"] = headtail_str(recovery_total)
        entry["(masked - recovery)"]   = headtail_str(masked_minus_recovery)
        entry["loss"] = float(round(loss, 6))
        entry["accuracy"] = float(round(acc, 6))
        entry["server global weights"] = headtail_str(_flatten(accum))
        self._dump(server_round, entry)

        self.U1_prev = set(U2)  # survivors của vòng này

        aggregated = ndarrays_to_parameters([a for a in accum])
        return aggregated, {}

def _init_params():
    model = SimpleMLP()
    nds = [v.detach().cpu().numpy() for _, v in model.state_dict().items()]
    return ndarrays_to_parameters(nds)

def main():
    strat = P1FedAvg(
        initial_parameters=_init_params(),
        min_fit_clients=T,
        min_evaluate_clients=T,
        min_available_clients=T,
        accept_failures=True,
    )
    print(f"[P1-Server] start 127.0.0.1:8081, N={N}, t={T}, rounds={ROUNDS}")
    fl.server.start_server(
        server_address="127.0.0.1:8081",
        strategy=strat,
        config=ServerConfig(num_rounds=ROUNDS, round_timeout=None),
    )

if __name__ == "__main__":
    main()
