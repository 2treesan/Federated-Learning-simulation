# server.py
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

# Dọn registry rank cũ mỗi lần chạy lại
try:
    if os.path.exists("p0_registry.json"):
        os.remove("p0_registry.json")
except Exception:
    pass

# ===== JSON options =====
JSON_NDIGITS = int(os.environ.get("P0_JSON_NDIGITS", "6"))  # làm tròn số
JSON_SPLIT_PER_EPOCH = os.environ.get("P0_JSON_SPLIT", "0") == "1"  # ghi mỗi epoch 1 file?
JSON_HEAD = int(os.environ.get("P0_JSON_HEAD", "5"))  # số phần tử đầu
JSON_TAIL = int(os.environ.get("P0_JSON_TAIL", "5"))  # số phần tử cuối

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
    assert len(state_keys) == len(nds), "Số lượng tensors không khớp"
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
    """Trích 5 số đầu + '...' + 5 số cuối (hoặc ít hơn nếu mảng ngắn)."""
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
    return head_list + ["..."] + tail_list  # vẫn là JSON hợp lệ

class P0FedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.json_path = "protocol0.json"
        self.protocol_log = {}
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
        ep_path = f"protocol0_epoch_{server_round}.json"
        with open(ep_path, "w", encoding="utf-8") as f:
            json.dump(epoch_entry, f, ensure_ascii=False, indent=2, sort_keys=True)

    def aggregate_fit(self, server_round, results, failures):
        print(f"\n[Server] Nhận {len(results)} kết quả fit ở Round {server_round}:")
        client_masked_nds = []
        client_unmasked_flat = {}
        client_masked_flat = {}

        for client_proxy, fit_res in results:
            cid = getattr(client_proxy, "cid", "unknown")
            nds = parameters_to_ndarrays(fit_res.parameters)  # MASKED
            print(f"  • From client {cid}: preview(masked) = {_preview(nds)} (n={fit_res.num_examples})")
            client_masked_nds.append(nds)

            metrics = fit_res.metrics or {}
            rank_str = metrics.get("rank", None)
            if rank_str is None:
                raise RuntimeError("Client không gửi kèm 'rank' trong metrics.")
            rank = int(rank_str)

            unmasked_flat_str = metrics.get("unmasked_flat", "[]")
            masked_flat_str = metrics.get("masked_flat", "[]")
            uflat = np.array(json.loads(unmasked_flat_str), dtype=np.float32)
            mflat = np.array(json.loads(masked_flat_str), dtype=np.float32)
            client_unmasked_flat[rank] = uflat
            client_masked_flat[rank] = mflat

        # Tổng (MASKED) tính từ parameters
        masked_total_flat = None
        if client_masked_nds:
            stacked = [_flatten(nds) for nds in client_masked_nds]
            masked_total_flat = np.sum(np.stack(stacked, axis=0), axis=0).astype(np.float32)
            k = min(6, masked_total_flat.size)
            with np.printoptions(precision=5, suppress=True):
                print(f"  • Tổng element-wise (preview): {np.array2string(masked_total_flat[:k], separator=', ')}")

        # Tổng (UNMASKED) tính từ metrics
        unmasked_total_flat = None
        if client_unmasked_flat:
            arrs = [client_unmasked_flat[r] for r in sorted(client_unmasked_flat.keys())]
            unmasked_total_flat = np.sum(np.stack(arrs, axis=0), axis=0).astype(np.float32)

        # FedAvg bình quân gia quyền
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

        # Đánh giá trên server (full test)
        loss, acc = _evaluate_on_server(accum)
        print(f"  • Server Eval => loss={loss:.6f}, acc={acc:.4f}\n")

        # ===== Ghi JSON (đẹp + rút gọn head-tail) =====
        epoch_key = f"epoch {server_round}"  # đổi format theo yêu cầu
        epoch_entry = {}

        # client0/1/2 unmasked & masked (head-tail)
        for r in sorted(client_unmasked_flat.keys()):
            epoch_entry[f"client {r} (unmasked)"] = _head_tail(client_unmasked_flat[r])
        for r in sorted(client_masked_flat.keys()):
            epoch_entry[f"client {r} (masked)"] = _head_tail(client_masked_flat[r])

        # tổng (unmasked & masked)
        epoch_entry["sum (unmasked)"] = _head_tail(unmasked_total_flat)
        epoch_entry["sum (masked)"]   = _head_tail(masked_total_flat)

        # loss & acc của server (làm tròn)
        epoch_entry["loss"] = float(round(loss, JSON_NDIGITS if JSON_NDIGITS else 6))
        epoch_entry["accuracy"] = float(round(acc, JSON_NDIGITS if JSON_NDIGITS else 6))

        # bộ trọng số toàn cục gửi về (flatten head-tail)
        epoch_entry["server global weights"] = _head_tail(_flatten(accum))

        # Lưu vào log tổng hợp và ghi ra file
        self.protocol_log[epoch_key] = epoch_entry
        self._dump_epoch_json(server_round, epoch_entry)
        self._dump_main_json()

        aggregated = ndarrays_to_parameters([a for a in accum])
        return aggregated, {}

def main():
    initial_parameters = get_initial_parameters()

    strategy = P0FedAvg(
        initial_parameters=initial_parameters,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    print("[Server] Khởi động gRPC server (compat) tại 127.0.0.1:8080 ...", flush=True)
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=ServerConfig(num_rounds=5, round_timeout=None),
    )
    print("[Server] Đã dừng server.", flush=True)

if __name__ == "__main__":
    main()
