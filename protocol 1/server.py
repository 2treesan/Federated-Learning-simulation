# server_p1.py
import os, json, collections
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import numpy as np, torch
from torch import nn
from model import SimpleMLP, get_test_loader
from protocol1 import (
    pair_seed, mask_vec_from_seed, 
    headtail_str
)
from shamir import reconstruct_secret, P as PRIME

N = int(os.environ.get("P1_NCLIENTS", "5"))
T = int(os.environ.get("P1_T", str((N // 2) + 1)))

JSON_HEAD = int(os.environ.get("P1_JSON_HEAD", "5"))
JSON_TAIL = int(os.environ.get("P1_JSON_TAIL", "5"))
JSON_NDIG = int(os.environ.get("P1_JSON_NDIGITS", "6"))

# dọn rank-registry
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

def _evaluate_on_server(nds):
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
        # pools
        self.shares_pool = collections.defaultdict(  # round_id ->
            lambda: collections.defaultdict(          # owner (rank) ->
                lambda: collections.defaultdict(dict) # pair_key "a-b" -> {recipient_rank: y}
            )
        )
        self.U1_prev = set(range(N))  # U1 của vòng trước; vòng 1 coi như đủ
        self.last_round = 0
        if os.path.exists(self.json_path):
            os.remove(self.json_path)

    # đồng bộ round + gửi U1_prev cho client
    def configure_fit(self, server_round, parameters, client_manager):
        ins = super().configure_fit(server_round, parameters, client_manager)
        for _, fitins in ins:
            cfg = dict(fitins.config or {})
            cfg["round_id"] = str(server_round)
            cfg["U1_prev"]  = json.dumps(sorted(list(self.U1_prev)))
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

        # ======= Thu shares (pha 1 hiện tại) + updates (pha 2 dựa U1_prev) =======
        for client_proxy, fit_res in results:
            cid = getattr(client_proxy, "cid", "unknown")
            nds = parameters_to_ndarrays(fit_res.parameters)  # masked
            print(f"  • From {cid}: preview(masked) = {headtail_str(_flatten(nds))}")
            masked_nds_list.append(nds); weights.append(fit_res.num_examples)

            m = fit_res.metrics or {}
            rank = int(m.get("rank", "-1")); survivors.add(rank)

            uf = np.array(json.loads(m.get("unmasked_flat", "[]")), dtype=np.float32)
            mf = np.array(json.loads(m.get("masked_flat", "[]")), dtype=np.float32)
            unmasked_flat[rank] = uf; masked_flat[rank] = mf

            # shares_out: { j: { recipient: y } } do OWNER=rank phát
            shares_out = json.loads(m.get("shares_out", "{}"))
            for j_str, per_rec in shares_out.items():
                j = int(j_str)
                a, b = (min(rank, j), max(rank, j))
                pair_key = f"{a}-{b}"
                for rcpt_str, y in per_rec.items():
                    rcpt = int(rcpt_str)
                    self.shares_pool[server_round][rank][pair_key][rcpt] = int(y)

        U2 = sorted(list(survivors))
        print(f"  • Survivors(U2)={U2}; U1_prev={sorted(list(self.U1_prev))}")
        dropouts = sorted(list(self.U1_prev - survivors))  # người rơi so với U1_prev

        # ======= Kiểm chứng tổng masked/unmasked và khôi phục bằng shares của "người rơi" =======
        masked_total = None
        if masked_nds_list:
            stacked = [_flatten(x) for x in masked_nds_list]
            masked_total = np.sum(np.stack(stacked, 0), 0).astype(np.float32)

        unmasked_total = None
        if unmasked_flat:
            arr = [unmasked_flat[r] for r in U2]
            unmasked_total = np.sum(np.stack(arr, 0), 0).astype(np.float32)

        # chuẩn bị chiều dài vector
        L = masked_total.size if masked_total is not None else (unmasked_total.size if unmasked_total is not None else 0)

        # Tính recovery từ các user RƠI: với mỗi v∈dropouts, reconstruct s_{u,v} (chính xác hơn: s_pair của (u,v) từ shares OWNER=v)
        recovery_total = np.zeros(L, dtype=np.float32)
        for v in dropouts:
            owner = v  # theo paper: dùng secrets **của người rơi**
            owner_shares = self.shares_pool.get(server_round, {}).get(owner, {})
            for u in U2:
                a, b = (min(u, v), max(u, v))
                pair_key = f"{a}-{b}"
                rcpts = owner_shares.get(pair_key, {})
                # lấy ≥T shares từ những người còn sống (hoặc bất kỳ, demo không mã hoá)
                avail = []
                for rcpt, y in rcpts.items():
                    if rcpt in U2 or True:   # demo: cho phép dùng mọi share đã nhận
                        avail.append((rcpt + 1, int(y)))  # x = rcpt+1
                if len(avail) < T:
                    continue  # không đủ share để reconstruct
                shares_subset = avail[:T]
                s_pair = reconstruct_secret(shares_subset, PRIME)
                base = mask_vec_from_seed(L, s_pair)
                # cần p_{u,v} => đổi dấu so với p_{v,u} nếu thứ tự khác
                sign = +1.0 if u < v else -1.0
                recovery_total += sign * base

        masked_minus_recovery = masked_total - recovery_total if masked_total is not None else None
        if masked_minus_recovery is not None and unmasked_total is not None:
            err = float(np.linalg.norm(masked_minus_recovery - unmasked_total))
            print(f"  • ||(masked - recovery) - unmasked||_2 = {err:.6e}")

        # ======= FedAvg TRÊN dữ liệu đã unmask từng client =======
        # Tạo correction riêng cho từng client u: sum_{v in dropouts} p_{u,v}
        corr_per_client = {u: np.zeros(L, dtype=np.float32) for u in U2}
        for v in dropouts:
            owner_shares = self.shares_pool.get(server_round, {}).get(v, {})
            for u in U2:
                a, b = (min(u, v), max(u, v))
                pair_key = f"{a}-{b}"
                rcpts = owner_shares.get(pair_key, {})
                avail = [(rcpt + 1, int(y)) for rcpt, y in rcpts.items()]
                if len(avail) < T:
                    continue
                s_pair = reconstruct_secret(avail[:T], PRIME)
                base = mask_vec_from_seed(L, s_pair)
                sign = +1.0 if u < v else -1.0
                corr_per_client[u] += sign * base

        corrected_nds_list = []
        for (nds, w), u in zip(masked_nds_list, U2):
            corr = corr_per_client[u]
            parts = []
            flat = _flatten(nds)
            flat_corr = flat - corr
            # unflatten theo nds hiện có
            i = 0
            for arr in nds:
                sz = arr.size
                parts.append(flat_corr[i:i+sz].reshape(arr.shape).astype(np.float32))
                i += sz
            corrected_nds_list.append((parts, w))

        # FedAvg
        totw = float(sum(w for _, w in corrected_nds_list)) or 1.0
        accum = None
        for nds, w in corrected_nds_list:
            if accum is None:
                accum = [a.astype(np.float32) * (w / totw) for a in nds]
            else:
                for i in range(len(nds)):
                    accum[i] += nds[i].astype(np.float32) * (w / totw)

        print("  • global preview:", headtail_str(_flatten(accum)))
        loss, acc = _evaluate_on_server(accum)
        print(f"  • Server Eval: loss={loss:.6f}, acc={acc:.4f}")

        # JSON (1 dòng / list)
        entry = {}
        for r in U2:
            entry[f"client {r} (unmasked)"] = headtail_str(unmasked_flat[r], JSON_HEAD, JSON_TAIL, JSON_NDIG)
            entry[f"client {r} (masked)"]   = headtail_str(masked_flat[r], JSON_HEAD, JSON_TAIL, JSON_NDIG)
        entry["U1_prev"] = sorted(list(self.U1_prev))
        entry["dropouts"] = dropouts
        entry["sum (unmasked)"] = headtail_str(unmasked_total, JSON_HEAD, JSON_TAIL, JSON_NDIG)
        entry["sum (masked)"]   = headtail_str(masked_total, JSON_HEAD, JSON_TAIL, JSON_NDIG)
        entry["recovery sum (masked)"] = headtail_str(recovery_total, JSON_HEAD, JSON_TAIL, JSON_NDIG)
        entry["(masked - recovery)"]   = headtail_str(masked_minus_recovery, JSON_HEAD, JSON_TAIL, JSON_NDIG)
        entry["loss"] = float(round(loss, JSON_NDIG))
        entry["accuracy"] = float(round(acc, JSON_NDIG))
        entry["server global weights"] = headtail_str(_flatten(accum), JSON_HEAD, JSON_TAIL, JSON_NDIG)
        self._dump(server_round, entry)

        # cập nhật U1 cho vòng kế: chính là U2 của vòng này
        self.U1_prev = set(U2)

        aggregated = ndarrays_to_parameters([a for a in accum])
        return aggregated, {}

def get_initial_parameters():
    model = SimpleMLP()
    nds = [v.detach().cpu().numpy() for _, v in model.state_dict().items()]
    return ndarrays_to_parameters(nds)

def main():
    strat = P1FedAvg(
        initial_parameters=get_initial_parameters(),
        min_fit_clients=max(T, 3),        # cần ≥t để đảm bảo đúng
        min_evaluate_clients=max(T, 3),
        min_available_clients=max(T, 3),
    )
    print(f"[P1-Server] start 127.0.0.1:8081, N={N}, t={T}")
    fl.server.start_server(
        server_address="127.0.0.1:8081",
        strategy=strat,
        config=ServerConfig(num_rounds=5, round_timeout=None),
    )

if __name__ == "__main__":
    main()
