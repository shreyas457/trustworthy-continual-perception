from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import as_classification_dataset
from torchvision import transforms
from soda10m_dataset import SODA10MDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_pytorch = SODA10MDataset(root="../SSLAD-2D/labeled/", split="train", transform=transform)
val_pytorch   = SODA10MDataset(root="../SSLAD-2D/labeled/", split="val",   transform=transform)

train_avl = AvalancheDataset(train_pytorch)
val_avl   = AvalancheDataset(val_pytorch)

# Wrap with explicit targets passed in — bypasses auto-detection entirely
train_avl = as_classification_dataset(train_pytorch)
val_avl = as_classification_dataset(val_pytorch)

# Split into experiences by subsets (e.g., different driving conditions)
# Here we just use a single experience as a baseline
benchmark = dataset_benchmark(
    train_datasets=[train_avl],
    test_datasets=[val_avl]
)

# Access experiences
for exp in benchmark.train_stream:
    print(f"Experience {exp.current_experience}: {len(exp.dataset)} samples")


