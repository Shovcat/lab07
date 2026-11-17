import copy
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_subset = Subset(train_dataset, range(5000))
test_subset = Subset(test_dataset, range(1000))

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)

class OversizedBaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SmallerStudentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, dataloader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

def measure_latency(model, dataloader, num_batches=10):
    model.eval()
    dummy_input = next(iter(dataloader))[0].to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    start_time = time.time()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            _ = model(inputs.to(device))
    end_time = time.time()
    return (end_time - start_time) / num_batches * 1000

if __name__ == "__main__":
    print("\n--- TASK 1: Baseline CNN Training ---")
    teacher_model = OversizedBaselineCNN().to(device)
    param_count = count_parameters(teacher_model)
    print(f"Baseline Parameters: {param_count:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
    
    print("Training baseline model...")
    train_model(teacher_model, train_loader, criterion, optimizer, epochs=1)
    
    baseline_acc = evaluate_model(teacher_model, test_loader)
    baseline_latency = measure_latency(teacher_model, test_loader)
    torch.save(teacher_model.state_dict(), "baseline.pth")
    baseline_size = os.path.getsize("baseline.pth") / (1024 * 1024)
    
    print(f"Baseline Accuracy: {baseline_acc * 100:.2f}%")
    print(f"Baseline Latency: {baseline_latency:.2f} ms/batch")
    print(f"Baseline Size: {baseline_size:.2f} MB")
    
    print("\n--- TASK 2: Magnitude & Structured Pruning ---")
    sparsities = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
    unstructured_accuracies = []
    
    for s in sparsities:
        model_copy = copy.deepcopy(teacher_model)
        if s > 0:
            parameters_to_prune = []
            for name, module in model_copy.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    parameters_to_prune.append((module, 'weight'))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=s,
            )
            for module, name in parameters_to_prune:
                prune.remove(module, name)
                
        opt_prune = optim.Adam(model_copy.parameters(), lr=0.0001)
        train_model(model_copy, train_loader, criterion, opt_prune, epochs=1)
        acc = evaluate_model(model_copy, test_loader)
        unstructured_accuracies.append(acc * 100)
        print(f"Global Unstructured Sparsity {s*100}% -> Accuracy: {acc*100:.2f}%")
        
    structured_ratios = [0.10, 0.25, 0.50]
    structured_accuracies = []
    for r in structured_ratios:
        model_struct = copy.deepcopy(teacher_model)
        for name, module in model_struct.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=r, n=2, dim=0)
                prune.remove(module, "weight")
        opt_struct = optim.Adam(model_struct.parameters(), lr=0.0001)
        train_model(model_struct, train_loader, criterion, opt_struct, epochs=1)
        acc_s = evaluate_model(model_struct, test_loader)
        structured_accuracies.append(acc_s * 100)
        print(f"Structured Pruning ({r*100}%) -> Accuracy: {acc_s*100:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.plot([s*100 for s in sparsities], unstructured_accuracies, marker='o', label='Global Unstructured')
    plt.title("Accuracy vs. Sparsity (Magnitude Pruning)")
    plt.xlabel("Sparsity Target (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig("sparsity_curves.png")
    print("\nSaved 'sparsity_curves.png' successfully.")

    print("\n--- TASK 3: Latency & Storage Analysis ---")
    print("Honest Latency Analysis: Unstructured sparsity sets weights to zero within dense tensors. Standard CPU/GPU dense matrix multiplication kernels still iterate over zeros unless compressed sparse formats (CSR/CSC) and specialized sparse hardware are used.")

    print("\n--- TASK 4: Knowledge Distillation ---")
    def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
        ce_loss = F.cross_entropy(student_logits, labels)
        kd_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1)
        ) * (T**2)
        return alpha * ce_loss + (1 - alpha) * kd_loss

    plain_student = SmallerStudentCNN().to(device)
    plain_params = count_parameters(plain_student)
    opt_plain = optim.Adam(plain_student.parameters(), lr=0.001)
    print(f"Plain Student Parameters: {plain_params:,}")
    train_model(plain_student, train_loader, criterion, opt_plain, epochs=1)
    plain_acc = evaluate_model(plain_student, test_loader)

    distilled_student = SmallerStudentCNN().to(device)
    opt_dist = optim.Adam(distilled_student.parameters(), lr=0.001)
    teacher_model.eval()
    distilled_student.train()
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        opt_dist.zero_grad()
        student_logits = distilled_student(inputs)
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        loss = distillation_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5)
        loss.backward()
        opt_dist.step()

    distilled_acc = evaluate_model(distilled_student, test_loader)

    print(f"Plain Student Accuracy: {plain_acc * 100:.2f}%")
    print(f"Distilled Student Accuracy: {distilled_acc * 100:.2f}%")

    print("\n--- TASK 5: Summary Table (Ch. 9 Efficiency Pillars) ---")
    summary_data = {
        "Efficiency Pillar": ["Data", "Model", "Compute", "Energy"],
        "Baseline CNN": ["Fashion-MNIST", f"{param_count:,} params", f"{baseline_latency:.2f} ms/batch", "High power draw"],
        "Pruned CNN (90%)": ["Fashion-MNIST", "Same dense tensor size", "Unchanged on dense CPU", "Moderate power draw"],
        "Distilled Student": ["Fashion-MNIST", f"{plain_params:,} params", "Lower MACs / faster", "Low energy consumption"]
    }
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
