# server.py
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
import torch
from model import SimpleMLP  # import cùng model.py bạn đang dùng
from flwr.common import ndarrays_to_parameters

def get_initial_parameters():
    # Tạo model giống client (same architecture)
    model = SimpleMLP()
    # Lấy state_dict và convert tensor -> numpy arrays (ndarrays)
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    # Chuyển ndarray list -> flower.Parameters
    return ndarrays_to_parameters(ndarrays)

def weighted_average(metrics):
    total_examples = sum(num for num, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}
    acc = sum(num * m["accuracy"] for num, m in metrics) / total_examples
    return {"accuracy": acc}

def main():
    # Lấy initial parameters từ server-side model
    initial_parameters = get_initial_parameters()

    strategy = FedAvg(
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=ServerConfig(num_rounds=3, round_timeout=None),
    )

if __name__ == "__main__":
    main()
