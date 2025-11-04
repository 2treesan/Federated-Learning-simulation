# client.py
import json
import flwr as fl
import torch
from torch import nn, optim
import numpy as np

from model import SimpleMLP, get_data_loaders
from protocol0 import Protocol0Masker, assign_rank, preview_params, l2_diff, flatten_list

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, num_clients: int = 3, rank: int = 0):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        self.num_clients = num_clients
        self.rank = rank
        self.masker = Protocol0Masker(rank=self.rank, num_clients=self.num_clients)
        self.local_round = 0
        print(f"[Client rank={self.rank}] Dataset shard size = {len(self.train_loader.dataset)}")

    def get_parameters(self):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()
        for _ in range(1):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE).long()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        unmasked = self.get_parameters()
        self.local_round = self.masker.next_round()
        masked = self.masker.mask_parameters(unmasked, round_id=self.local_round)

        print(f"[Client rank={self.rank} | Round {self.local_round}]")
        print(f"  • Preview params (gốc):    {preview_params(unmasked)}")
        print(f"  • Preview params (masked): {preview_params(masked)}")
        print(f"  • ||masked - gốc||_2 = {l2_diff(masked, unmasked):.6f}")

        flat_unmasked = flatten_list(unmasked).astype(np.float32).tolist()
        flat_masked = flatten_list(masked).astype(np.float32).tolist()
        metrics = {
            "rank": str(self.rank),
            "unmasked_flat": json.dumps(flat_unmasked),
            "masked_flat": json.dumps(flat_masked),
        }

        num_examples = len(self.train_loader.dataset)
        return masked, num_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        loss = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE).long()
                output = self.model(data)
                loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    num_clients = 3
    rank = assign_rank(num_clients=num_clients)
    model = SimpleMLP().to(DEVICE)
    train_loader, test_loader = get_data_loaders(rank=rank, num_clients=num_clients)
    client = FlowerClient(model, train_loader, test_loader, num_clients=num_clients, rank=rank)

    # DÙNG API MỚI (compat) như cảnh báo gợi ý
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )
