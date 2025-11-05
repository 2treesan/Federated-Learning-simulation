# client.py  (Protocol 1)
import os, json
import flwr as fl
import torch
from torch import nn, optim
import numpy as np

from model import SimpleMLP, get_data_loaders
from protocol1 import (
    assign_rank, pair_base_seed, seed_from_base, mask_vec_from_seed,
    flatten_params, unflatten_params, headtail_str, PRIME
)
from shamir import det_share_secret

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClientP1(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, n_hint: int, rank: int, t_threshold: int):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.rank = rank
        self.n_hint = n_hint
        self.t = t_threshold
        print(f"[P1-Client rank={self.rank}] full train size={len(self.train_loader.dataset)}")

    def get_parameters(self):
        return [v.detach().cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, params):
        state = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params)}
        self.model.load_state_dict(state, strict=True)

    def _one_epoch(self):
        self.model.train()
        for x, y in self.train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        round_id = int(config.get("round_id", 1))
        U1_prev = set(json.loads(config.get("U1_prev", "[]")))
        N_total = int(config.get("num_clients_total", max(self.n_hint, len(U1_prev) or 1)))
        roster_all = set(json.loads(config.get("roster_all", json.dumps(list(range(N_total))))))

        self._one_epoch()

        unmasked = self.get_parameters()
        flat, shapes, dtypes = flatten_params(unmasked)
        L = flat.size

        # === PHASE SHARE: dùng det_share_secret cố định cho mỗi cặp (u,v) ===
        shares_out = {}
        for j in sorted(roster_all):
            if j == self.rank:
                continue
            s_base = pair_base_seed(self.rank, j)
            shares = det_share_secret(s_base, n=N_total, t=self.t, salt=s_base)  # cố định theo cặp
            shares_out[str(j)] = {str(rcpt-1): int(y) for rcpt, y in shares}

        # === PHASE MASK: chỉ mask nếu self ∈ U1_prev (client mới -> không mask) ===
        total_mask = np.zeros(L, dtype=np.float32)
        if self.rank in U1_prev:
            for v in U1_prev:
                if v == self.rank:
                    continue
                s_base = pair_base_seed(self.rank, v)
                s_round = seed_from_base(s_base, round_id)
                base = mask_vec_from_seed(L, s_round)
                sign = +1.0 if self.rank < v else -1.0
                total_mask += sign * base

        masked_flat = flat + total_mask
        masked = unflatten_params(masked_flat, shapes, dtypes)

        print(f"[P1-Client {self.rank} | r={round_id}] U1_prev={sorted(list(U1_prev))} mask_on={self.rank in U1_prev}")
        print("  preview(x):", headtail_str(flat))
        print("  preview(y):", headtail_str(masked_flat))

        metrics = {
            "rank": str(self.rank),
            "unmasked_flat": json.dumps(flat.astype(np.float32).tolist()),
            "masked_flat": json.dumps(masked_flat.astype(np.float32).tolist()),
            "shares_out": json.dumps(shares_out),
        }
        return masked, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        tot, corr, cnt = 0.0, 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                out = self.model(x)
                loss = self.criterion(out, y)
                tot += loss.item() * x.size(0)
                pred = out.argmax(1, keepdim=True)
                corr += pred.eq(y.view_as(pred)).sum().item()
                cnt += x.size(0)
        return float(tot/cnt), cnt, {"accuracy": float(corr/cnt)}

if __name__ == "__main__":
    N = int(os.environ.get("P1_NCLIENTS", "5"))
    T = int(os.environ.get("P1_T", str((N // 2) + 1)))
    rank = assign_rank(N)

    model = SimpleMLP().to(DEVICE)
    train_loader, test_loader = get_data_loaders()

    client = FlowerClientP1(model, train_loader, test_loader, n_hint=N, rank=rank, t_threshold=T)
    fl.client.start_client(server_address="127.0.0.1:8081", client=client.to_client())
