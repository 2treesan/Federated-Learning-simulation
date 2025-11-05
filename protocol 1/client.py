# client_p1.py
import json
import flwr as fl
import torch
from torch import nn, optim
import numpy as np

from model import SimpleMLP, get_data_loaders
from protocol1 import (
    assign_rank, pair_seed, mask_vec_from_seed,
    flatten_params, unflatten_params, headtail_str
)
from shamir import share_secret, P as PRIME

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClientP1(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, nclients: int, rank: int, t_threshold: int):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.n = nclients
        self.rank = rank
        self.t = t_threshold
        print(f"[P1-Client rank={self.rank}] full train size={len(self.train_loader.dataset)}")

    def get_parameters(self):
        return [v.detach().cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, params):
        state = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params)}
        self.model.load_state_dict(state, strict=True)

    def _one_local_epoch(self):
        self.model.train()
        for data, target in self.train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE).long()
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        round_id = int(config.get("round_id", 1))
        U1_prev = set(json.loads(config.get("U1_prev", "[]")))  # U1 của vòng trước
        # Train 1 epoch (demo)
        self._one_local_epoch()

        # Lấy local params
        unmasked = self.get_parameters()
        flat, shapes, dtypes = flatten_params(unmasked)
        L = flat.size

        # === PHA 1 (vòng hiện tại): chia sẻ Shamir seeds cho mọi cặp (self, j)
        shares_out = {}  # { str(j): { str(recipient): y } }
        for j in range(self.n):
            if j == self.rank: 
                continue
            s_pair = pair_seed(self.rank, j, round_id)  # seed chung cho (min,max,round)
            shares = share_secret(s_pair, n=self.n, t=self.t)
            shares_out[str(j)] = {str(rcpt-1): int(y) for rcpt, y in shares}  # rcpt index = x-1

        # === PHA 2: mask chỉ theo U1_prev (tập sống vòng trước)
        total_mask = np.zeros(L, dtype=np.float32)
        for v in U1_prev:
            if v == self.rank: 
                continue
            s_pair = pair_seed(self.rank, v, round_id)
            base = mask_vec_from_seed(L, s_pair)
            sign = +1.0 if self.rank < v else -1.0
            total_mask += sign * base

        masked_flat = flat + total_mask
        masked = unflatten_params(masked_flat, shapes, dtypes)

        # Log client
        print(f"[P1-Client {self.rank} | r={round_id}] y= x + masks(U1_prev={sorted(list(U1_prev))})")
        print("  preview(x):", headtail_str(flat))
        print("  preview(y):", headtail_str(masked_flat))

        # Metrics gửi lên
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
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE).long()
                out = self.model(data)
                loss += self.criterion(out, target).item() * data.size(0)
                pred = out.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        return float(loss/total), total, {"accuracy": float(correct/total)}

if __name__ == "__main__":
    N = 5
    T = (N // 2) + 1  # t > n/2
    rank = assign_rank(N)
    model = SimpleMLP().to(DEVICE)
    train_loader, test_loader = get_data_loaders()
    client = FlowerClientP1(model, train_loader, test_loader, nclients=N, rank=rank, t_threshold=T)
    fl.client.start_client(server_address="127.0.0.1:8081", client=client.to_client())
