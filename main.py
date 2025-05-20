import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from collections import namedtuple

class HASYv2(Dataset):
    def __init__(self, hf_dataset, cut=None, transform=None):
        self.cut = cut if cut else len(hf_dataset)
        self.dataset = hf_dataset.select(range(cut))
        self.transform = transform
        print(f"loaded dataset of size: {len(self.dataset)}")
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        label = self.dataset[index]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

class MLP(nn.Module):
    def __init__(self, sizes=[32*32, 784, 369]):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(),
            nn.Linear(sizes[1], sizes[1]), # 2
            nn.ReLU(),
            nn.Linear(sizes[1], sizes[2]),
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(module.bias)        

    def forward(self, x, y=None):
        logits = self.mlp(x.view(x.shape[0], -1))
        loss = F.cross_entropy(logits, y) if y is not None else None
        return loss, logits

def train(model, train_dataset, test_dataset, optimizer, n_epochs, batch_size, w_save_key):
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    n_batches = len(train_dataset) // batch_size
    print(f"training with: {n_epochs} epochs | {batch_size} batch size | {n_batches} batches/epoch")

    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        for img, label in loader:
            img, label = img.to(device, dtype), label.to(device)
            _, logits = model(img, label)
            _, pred = torch.max(logits.data, dim=1)
            total += label.shape[0]
            correct += (pred == label).sum().item()
        return correct / total
    
    losses = []
    train_accs = [evaluate(train_loader)]
    test_accs = [evaluate(test_loader)]
    tracked_ws = [model.state_dict()[w_save_key].detach().to('cpu', torch.float32).clone()]
    print(f"initial state: train_acc:{train_accs[-1]:2.2%} | test_acc:{test_accs[-1]:2.2%}")
    for epoch in range(n_epochs):
        model.train()
        for batch_idx, (img, label) in enumerate(train_loader):
            if (batch_idx+1) % 32 == 0:
                tracked_ws.append(model.state_dict()[w_save_key].detach().to('cpu', torch.float32).clone())
            img, label = img.to(device, dtype), label.to(device)
            model.zero_grad()
            loss, _ = model(img, label)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        train_accs.append(evaluate(train_loader))
        test_accs.append(evaluate(test_loader))
        avg_loss = sum(losses[-n_batches:]) / n_batches
        print(f"epoch: {epoch:<3} | avg_loss: {avg_loss:.4f} | train_acc: {train_accs[-1]:2.2%} | test_acc: {test_accs[-1]:2.2%}")
    return namedtuple("train_result", ["losses", "train_accs", "test_accs", "tracked_ws"])(losses, train_accs, test_accs, tracked_ws)

def draw_a(size):
    # taken from  https://gist.github.com/ssnl/8a67601b309a4b64ba4363b5e66fb1c8
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arial.ttf", size) # download https://github.com/matomo-org/travis-scripts/blob/master/fonts/Arial.ttf
    draw.text((0, 0), 'a', font=font, fill=0)
    matrix = np.array(img)
    matrix = (matrix < 128).astype(int)
    return matrix

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"running with: {device} | {dtype}")
    
    # 369 classes
    train_cut = 131072 # 2**17 /151,241
    test_cut = 4096 # 16,992
    transform = transforms.Compose([transforms.ToTensor()])
    hf_train_dataset = load_dataset("randall-lab/hasy-v2", split="train", trust_remote_code=True)
    hf_test_dataset = load_dataset("randall-lab/hasy-v2", split="test", trust_remote_code=True)
    train_dataset = HASYv2(hf_train_dataset, cut=train_cut, transform=transform)
    test_dataset = HASYv2(hf_test_dataset, cut=test_cut, transform=transform)

    model = MLP([32*32, 784, 369]).to(device, dtype)
    compile = False
    target_layer_idx = 2
    weight_save_path = "a_tracked_ws.pt"

    batch_size = 128
    n_epochs = 5 # Increase this to see more change in the weights
    lr = 1e-1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # compiling the model changes the names of layers in state_dict
    if compile: # doesn't work with num_workers>1
        model = torch.compile(model)
        target_layer = model._orig_mod.mlp[target_layer_idx]
        original_weight = target_layer.weight.data.clone().detach()
        w_save_key = f"_orig_mod.mlp.{target_layer_idx}.weight"
    else:
        target_layer = model.mlp[target_layer_idx]
        original_weight = target_layer.weight.data.clone().detach()
        w_save_key = f"mlp.{target_layer_idx}.weight"

    a_weight = draw_a(52 * 28-20)[:-118][-28*28:, :28*28]
    a_mask = torch.from_numpy(a_weight).to(device, dtype=torch.bool)
    #plt.imshow(a_weight); plt.show()

    original_sd = {k: v.clone().detach() for k, v in model.state_dict().items()}  
    target_layer.weight.data[~a_mask] = 0.0
    modified_sd = {k: v.clone().detach() for k, v in model.state_dict().items()}

    # visualize original and modified weight matrix
    #viz_w_fn = lambda w: plt.imshow(w.to('cpu', torch.float32).data) and plt.show()
    #viz_w_fn(modified_sd[w_save_key])
    #viz_w_fn(original_sd[w_save_key])

    # TRAINING
    print("-"* 25 + "painted weight" + "-"* 25)
    a_trained = train(model, train_dataset, test_dataset, optimizer, n_epochs, batch_size, w_save_key)
    torch.save(a_trained.tracked_ws, weight_save_path)
    print(f"training complete, saved tracked weights to {weight_save_path}")

    """
    # compare with randomly initialized weights
    print("-"* 25 + "random weights" + "-"* 25)
    model.load_state_dict(original_sd)
    random_trained = train(model, train_dataset, test_dataset, optimizer, n_epochs, batch_size, w_save_key)
    #torch.save(random_trained.tracked_ws, "random_tracked_ws.pt")

    # VISUALIZATIONS
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(a_trained.train_accs, label="a")
    axes[0].plot(random_trained.train_accs, label="random")
    axes[0].set_title('train_accs'); axes[0].legend()

    axes[1].plot(a_trained.test_accs, label="a")
    axes[1].plot(random_trained.test_accs, label="random")
    axes[1].set_title('test_accs'); axes[1].legend()

    axes[2].plot(a_trained.losses)
    axes[2].set_title('train_loss')

    plt.savefig("evals.png")
    plt.show()
    """