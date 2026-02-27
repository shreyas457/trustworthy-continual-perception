from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
import torch.nn as nn
from torchvision.models import resnet18
from dataset_build import benchmark
import torch


model = resnet18(num_classes=6)  # SODA10M has 6 categories

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    loggers=[InteractiveLogger()]
)

strategy = Naive(
    model, optimizer, criterion,
    train_mb_size=32,
    train_epochs=3,
    eval_mb_size=64,
    evaluator=eval_plugin
)

for exp in benchmark.train_stream:
    strategy.train(exp)
    strategy.eval(benchmark.test_stream)