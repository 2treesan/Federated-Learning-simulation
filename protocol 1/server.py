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

# ===== Cấu hình số client kỳ vọng =====
P1_NCLIENTS = int(os.environ.get("P1_NCLIENTS", "5"))

# ===== JSON options =====
JSON_NDIGITS = int(os.environ.get("P1_JSON_NDIGITS", "6"))     # làm tròn khi in
JSON_HEAD = int(os.environ.get("P1_JSON_HEAD", "5"))           # head size
JSON_TAIL = int(os.environ.get("P1_JSON_TAIL", "5"))           # tail size
JSON_SPLIT_PER_EPOCH = os.environ.get("P1_JSON_SPLIT", "0") == "1"

# Dọn registry rank cũ mỗi lần chạy lại (để 5 client xếp rank 0..4)
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

def _head_tail(arr: np.ndarray, head: int = JSON_HEAD, tail: int = JSON_TAIL, ndigits: int = JSON_NDIGITS):
    if arr is None:
        return []
    arr = np.asarray(arr, dtype=np.float32).flatten()
    if ndigits is not None:
        arr = np.round(arr, decimals=int(ndigits))
    n = arr.size
    if n <= head + tail:
        return arr.tolist()
    head_list = arr[:head].tolist()
    tail_list = arr[-tail:].tolist()
    return head_list + ["..."] + tail_list

class P1FedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.json_path = "protocol1.json"
        self.protocol_log = {}
        self.expected_alive = set(range(P1_NCLIENTS))  # round 1: giả định đủ N client
        self._ensure_clean_json()

    def _ensure_clean_json(self):
        try:
            if os.path.exists(self.json_path):
                os.remove(self.json_path)
        except Exception:
            pass

    def _dump_main_json(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.protocol_log, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _dump_epoch_json(self, server_round: int, epoch_entry: dict):
        if not JSON_SPLIT_PER_EPOCH:
            return
        ep_path = f"protocol1_epoch_{server_round}.json"
        with open(ep_path, "w", encoding="utf-8") as f:
            json.dump(epoch_entry, f, ensure_ascii=False, indent=2, sort_keys=True)

    def aggregate_fit(self, server_round, results, failures):
        print(f"\n[P1-Server] Nhận {len(results)} kết quả fit ở Round {server_round}:")
        client_masked_nds = []
        client_unmasked_flat = {}
        client_masked_flat = {}
        client_contribs = {}  # rank -> {peer_rank -> np.ndarray}

        received_ranks = set()

        for client_proxy, fit_res in results:
            cid = getattr(client_proxy, "cid", "unknown")
            nds = parameters_to_ndarrays(fit_res.parameters)  # MASKED
            print(f"  • From client {cid}: preview(masked) = {_preview(nds)} (n={fit_res.num_examples})")
            client_masked_nds.append(nds)

            metrics = fit_res.metrics or {}
            rank = int(metrics.get("rank", "-1"))
            if rank < 0:
                raise RuntimeError("Client không gửi 'rank' trong metrics.")
            received_ranks.add(rank)

            uflat = np.array(json.loads(metrics.get("unmasked_flat", "[]")), dtype=np.float32)
            mflat = np.array(json.loads(metrics.get("masked_flat", "[]")), dtype=np.float32)
            contribs_dict = json.loads(metrics.get("contribs", "{}"))

            client_unmasked_flat[rank] = uflat
            client_masked_flat[rank] = mflat

            per_peer = {}
            for k, v in contribs_dict.items():
                per_peer[int(k)] = np.array(v, dtype=np.float32)
            client_contribs[rank] = per_peer

        # Xác định dropouts so với kỳ vọng cho vòng này
        dropouts = sorted(list(self.expected_alive - received_ranks))
        survivors = sorted(list(received_ranks))
        print(f"  • Survivors: {survivors} | Dropouts: {dropouts}")

        # Tổng masked như nhận được
        masked_total_flat = None
        if client_masked_nds:
            stacked = [_flatten(nds) for nds in client_masked_nds]
            masked_total_flat = np.sum(np.stack(stacked, axis=0), axis=0).astype(np.float32)
            print(f"  • Tổng (masked) preview: {_head_tail(masked_total_flat)}")

        # Tổng unmasked (chỉ từ client còn sống)
        unmasked_total_flat = None
        if client_unmasked_flat:
            arrs = [client_unmasked_flat[r] for r in survivors]
            unmasked_total_flat = np.sum(np.stack(arrs, axis=0), axis=0).astype(np.float32)

        # Khôi phục phần mask của người rớt: sum_u contrib[u][v] với mọi v thuộc dropouts
        recovery_total = None
        if dropouts:
            acc = None
            for u in survivors:
                per_peer = client_contribs.get(u, {})
                # cộng tất cả contributions hướng tới các dropout
                for v in dropouts:
                    if v in per_peer:
                        vec = per_peer[v]
                        acc = vec.astype(np.float32) if acc is None else (acc + vec.astype(np.float32))
            recovery_total = acc if acc is not None else np.zeros_like(masked_total_flat, dtype=np.float32)
            print(f"  • Recovery (sum contribs tới dropouts) preview: {_head_tail(recovery_total)}")

        # Sau khi khử phần dropouts, masked_total -> nên trùng unmasked_total
        if masked_total_flat is not None and recovery_total is not None:
            masked_minus_recovery = masked_total_flat - recovery_total
            diff_norm = float(np.linalg.norm(masked_minus_recovery - unmasked_total_flat))
            print(f"  • ||(masked - recovery) - unmasked||_2 = {diff_norm:.6e}")
        else:
            masked_minus_recovery = masked_total_flat

        # FedAvg bình quân gia quyền từ dữ liệu nhận được (chỉ survivors)
        total_weight = float(sum(fr.num_examples for _, fr in results))
        accum = None
        for nds, (_, fr) in zip(client_masked_nds, results):
            w = fr.num_examples
            if accum is None:
                accum = [arr.astype(np.float32) * (w / total_weight) for arr in nds]
            else:
                for i in range(len(nds)):
                    accum[i] += nds[i].astype(np.float32) * (w / total_weight)

        print(f"  • Tham số toàn cục (preview) gửi về: {_preview(accum)}")

        # Đánh giá trên server
        loss, acc = _evaluate_on_server(accum)
        print(f"  • Server Eval => loss={loss:.6f}, acc={acc:.4f}\n")

        # ===== Ghi JSON gọn (head...tail) =====
        epoch_key = f"epoch {server_round}"
        entry = {}
        # per-client
        for r in survivors:
            entry[f"client {r} (unmasked)"] = _head_tail(client_unmasked_flat[r])
            entry[f"client {r} (masked)"]   = _head_tail(client_masked_flat[r])
        entry["dropouts"] = dropouts
        entry["sum (unmasked)"] = _head_tail(unmasked_total_flat)
        entry["sum (masked)"]   = _head_tail(masked_total_flat)
        if recovery_total is not None:
            entry["recovery sum (masked)"] = _head_tail(recovery_total)
            entry["(masked - recovery)"]   = _head_tail(masked_minus_recovery)
        entry["loss"] = float(round(loss, JSON_NDIGITS))
        entry["accuracy"] = float(round(acc, JSON_NDIGITS))
        entry["server global weights"] = _head_tail(_flatten(accum))

        self.protocol_log[epoch_key] = entry
        if JSON_SPLIT_PER_EPOCH:
            with open(f"protocol1_epoch_{server_round}.json", "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2, sort_keys=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.protocol_log, f, ensure_ascii=False, indent=2, sort_keys=True)

        # Cập nhật expected_alive cho vòng kế: chính là các survivors
        self.expected_alive = set(survivors)

        aggregated = ndarrays_to_parameters([a for a in accum])
        return aggregated, {}

def main():
    initial_parameters = get_initial_parameters()

    strategy = P1FedAvg(
        initial_parameters=initial_parameters,
        # quan trọng: cho phép chạy tiếp với >=3 client
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
