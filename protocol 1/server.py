# server_p1.py
import os
import json
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
from model import SimpleMLP, get_test_loader
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

import numpy as np
import torch
from torch import nn

# ===== tham số =====
P1_NCLIENTS = int(os.environ.get("P1_NCLIENTS", "5"))
JSON_NDIGITS = int(os.environ.get("P1_JSON_NDIGITS", "6"))
JSON_HEAD = int(os.environ.get("P1_JSON_HEAD", "5"))
JSON_TAIL = int(os.environ.get("P1_JSON_TAIL", "5"))
JSON_SPLIT_PER_EPOCH = os.environ.get("P1_JSON_SPLIT", "0") == "1"

# Dọn registry
try:
    if os.path.exists("p0_registry.json"):
        os.remove("p0_registry.json")
except Exception:
    pass

def get_initial_parameters():
    model = SimpleMLP()
    ndarrays = [val.detach().cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(ndarrays)

def _flatten(nds):
    return np.concatenate([a.reshape(-1).astype(np.float32, copy=False) for a in nds], axis=0)

def _preview(nds, k=6):
    vec = _flatten(nds)
    k = min(k, vec.size)
    with np.printoptions(precision=5, suppress=True):
        return np.array2string(vec[:k], separator=", ")

def _set_model_from_nds(model: SimpleMLP, nds):
    state_keys = list(model.state_dict().keys())
    assert len(state_keys) == len(nds)
    state_dict = {k: torch.tensor(v) for k, v in zip(state_keys, nds)}
    model.load_state_dict(state_dict, strict=True)

def _evaluate_on_server(nds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)
    _set_model_from_nds(model, nds)
    model.eval()
    test_loader = get_test_loader(test_batch_size=256)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device).long()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    return float(total_loss / total), float(correct / total)

# ===== JSON helpers: in 1 dòng/list =====
def _head_tail_str(array_like, head=JSON_HEAD, tail=JSON_TAIL, ndigits=JSON_NDIGITS):
    if array_like is None:
        return "[]"
    arr = np.asarray(array_like, dtype=np.float32).ravel()
    if ndigits is not None:
        arr = np.round(arr, decimals=int(ndigits))
    n = arr.size
    if n <= head + tail:
        elems = [f"{float(x):.{int(ndigits)}f}" for x in arr]
    else:
        head_list = [f"{float(x):.{int(ndigits)}f}" for x in arr[:head]]
        tail_list = [f"{float(x):.{int(ndigits)}f}" for x in arr[-tail:]]
        elems = head_list + ['"..."'] + tail_list  # giữ "..." là chuỗi
    return "[" + ", ".join(elems) + "]"

def _unflatten_like(vec_flat: np.ndarray, template_nds):
    out, idx = [], 0
    for a in template_nds:
        size = a.size
        part = vec_flat[idx: idx + size].reshape(a.shape).astype(np.float32)
        out.append(part)
        idx += size
    return out

class P1FedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.json_path = "protocol1.json"
        self.protocol_log = {}
        self.expected_alive = set(range(P1_NCLIENTS))
        try:
            if os.path.exists(self.json_path):
                os.remove(self.json_path)
        except Exception:
            pass

    # >>>>>>>>>>>>>>>>>>>>  QUAN TRỌNG: ĐỒNG BỘ ROUND TỪ SERVER  <<<<<<<<<<<<<<<<<<
    def configure_fit(self, server_round, parameters, client_manager):
        ins = super().configure_fit(server_round, parameters, client_manager)
        # gắn round_id (và tổng N) vào config gửi xuống tất cả clients
        for _, fitins in ins:
            cfg = dict(fitins.config) if fitins.config is not None else {}
            cfg["round_id"] = str(server_round)
            cfg["num_clients_total"] = str(P1_NCLIENTS)
            fitins.config = cfg
        return ins

    def _dump(self, server_round: int, entry: dict):
        self.protocol_log[f"epoch {server_round}"] = entry
        if JSON_SPLIT_PER_EPOCH:
            with open(f"protocol1_epoch_{server_round}.json", "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2, sort_keys=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.protocol_log, f, ensure_ascii=False, indent=2, sort_keys=True)

    def aggregate_fit(self, server_round, results, failures):
        print(f"\n[P1-Server] Nhận {len(results)} kết quả fit ở Round {server_round}:")
        # thu thập
        masked_nds_list = []
        per_client_weight = []
        unmasked_flat = {}
        masked_flat = {}
        contribs = {}
        survivors = set()

        for client_proxy, fit_res in results:
            cid = getattr(client_proxy, "cid", "unknown")
            nds = parameters_to_ndarrays(fit_res.parameters)  # masked
            print(f"  • From client {cid}: preview(masked) = {_preview(nds)} (n={fit_res.num_examples})")
            masked_nds_list.append(nds)
            per_client_weight.append(fit_res.num_examples)

            m = fit_res.metrics or {}
            rank = int(m.get("rank", "-1"))
            survivors.add(rank)
            unmasked_flat[rank] = np.array(json.loads(m.get("unmasked_flat", "[]")), dtype=np.float32)
            masked_flat[rank]   = np.array(json.loads(m.get("masked_flat", "[]")), dtype=np.float32)
            cdict = json.loads(m.get("contribs", "{}"))
            contribs[rank] = {int(k): np.array(v, dtype=np.float32) for k, v in cdict.items()}

        dropouts = sorted(list(self.expected_alive - survivors))
        survivors = sorted(list(survivors))
        print(f"  • Survivors: {survivors} | Dropouts: {dropouts}")

        # tổng masked (survivors)
        masked_total_flat = None
        if masked_nds_list:
            stacked = [_flatten(nds) for nds in masked_nds_list]
            masked_total_flat = np.sum(np.stack(stacked, axis=0), axis=0).astype(np.float32)

        # tổng unmasked (survivors) — để kiểm chứng
        unmasked_total_flat = None
        if unmasked_flat:
            arrs = [unmasked_flat[r] for r in survivors]
            unmasked_total_flat = np.sum(np.stack(arrs, axis=0), axis=0).astype(np.float32)

        # ======= UNMASK THEO TỪNG CLIENT trước FedAvg =======
        corrected_nds_list = []
        recovery_total = np.zeros_like(masked_total_flat, dtype=np.float32) if masked_total_flat is not None else None

        for nds, w, rank in zip(masked_nds_list, per_client_weight, survivors):
            # correction cho client này: sum_{v in dropouts} contribs[rank][v]
            corr_vec = None
            per_peer = contribs.get(rank, {})
            for v in dropouts:
                if v in per_peer:
                    vec = per_peer[v]
                    corr_vec = vec.astype(np.float32) if corr_vec is None else (corr_vec + vec.astype(np.float32))
            if corr_vec is None:
                corr_vec = np.zeros_like(_flatten(nds), dtype=np.float32)
            else:
                recovery_total += corr_vec

            # trừ correction khỏi ND arrays của client này
            corr_parts = _unflatten_like(corr_vec, nds)
            corrected = [a.astype(np.float32) - c.astype(np.float32) for a, c in zip(nds, corr_parts)]
            corrected_nds_list.append((corrected, w))

        # Kiểm chứng: (sum masked) - (sum recovery) ≈ (sum unmasked)
        if masked_total_flat is not None and recovery_total is not None and unmasked_total_flat is not None:
            masked_minus_recovery = masked_total_flat - recovery_total
            check = float(np.linalg.norm(masked_minus_recovery - unmasked_total_flat))
            print(f"  • ||(masked - recovery) - unmasked||_2 = {check:.6e}")

        # ======= FedAvg trên DỮ LIỆU ĐÃ UNMASK =======
        total_weight = float(sum(w for _, w in corrected_nds_list))
        accum = None
        for nds, w in corrected_nds_list:
            if accum is None:
                accum = [arr.astype(np.float32) * (w / total_weight) for arr in nds]
            else:
                for i in range(len(nds)):
                    accum[i] += nds[i].astype(np.float32) * (w / total_weight)

        print(f"  • Tham số toàn cục (preview) gửi về: {_preview(accum)}")
        loss, acc = _evaluate_on_server(accum)
        print(f"  • Server Eval => loss={loss:.6f}, acc={acc:.4f}\n")

        # ===== JSON: 1 dòng mỗi list =====
        entry = {}
        for r in survivors:
            entry[f"client {r} (unmasked)"] = _head_tail_str(unmasked_flat[r])
            entry[f"client {r} (masked)"]   = _head_tail_str(masked_flat[r])
        entry["dropouts"] = dropouts
        entry["sum (unmasked)"] = _head_tail_str(unmasked_total_flat)
        entry["sum (masked)"]   = _head_tail_str(masked_total_flat)
        if masked_total_flat is not None and recovery_total is not None:
            entry["recovery sum (masked)"] = _head_tail_str(recovery_total)
            entry["(masked - recovery)"]   = _head_tail_str(masked_total_flat - recovery_total)
        entry["loss"] = float(round(loss, JSON_NDIGITS))
        entry["accuracy"] = float(round(acc, JSON_NDIGITS))
        entry["server global weights"] = _head_tail_str(_flatten(accum))

        self._dump(server_round, entry)
        self.expected_alive = set(survivors)

        aggregated = ndarrays_to_parameters([a for a in accum])
        return aggregated, {}

def main():
    initial_parameters = get_initial_parameters()
    strategy = P1FedAvg(
        initial_parameters=initial_parameters,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )
    print("[P1-Server] Khởi động gRPC server (compat) tại 127.0.0.1:8081 ...", flush=True)
    fl.server.start_server(
        server_address="127.0.0.1:8081",
        strategy=strategy,
        config=ServerConfig(num_rounds=5, round_timeout=None),
    )
    print("[P1-Server] Đã dừng server.", flush=True)

if __name__ == "__main__":
    main()
