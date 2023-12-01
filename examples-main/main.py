import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * 12 * 12, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        return x

def train(model, input, target, criterion, optimizer):
    output = model(input)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def worker(model, input, target, criterion, optimizer, lock, process_id):
    torch.manual_seed(42)  # Ensure deterministic behavior
    with lock:
        print(f"Process {process_id}: Starting training.")
    for epoch in range(10):
        loss = train(model, input, target, criterion, optimizer)
        with lock:
            print(f"Process {process_id}: Epoch {epoch + 1}, Loss: {loss:.4f}")
    with lock:
        print(f"Process {process_id}: Training complete.")

def main():
    model = SimpleCNN()
    input = torch.randn(1, 1, 28, 28)
    target = torch.randint(0, 2, (1,), dtype=torch.long)  # Assuming binary classification
    epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_processes = 4  # You can adjust this based on your hardware capabilities
    lock = mp.Lock()

    processes = []
    for process_id in range(num_processes):
        p = mp.Process(target=worker, args=(model, input, target, criterion, optimizer, lock, process_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
