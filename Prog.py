import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d, Dropout
from pytorch_msssim import ssim
device = torch.device(&quot;cuda&quot; if torch.cuda.is_available() else &quot;cpu&quot;)
print(f&quot;Appareil utilisé : {device}&quot;)
image_path = &quot;C:\\TraitementImage\\imag1.jpg&quot;
image = cv2.imread(image_path)
if image is None:
raise FileNotFoundError(&quot;Image introuvable.&quot;)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, c = image.shape
def image_to_graph_color_with_pos(image):
h, w, c = image.shape
edges = []
node_features = []
for i in range(h):
for j in range(w):
pixel = image[i, j] / 255.0
pos = [i / h, j / w]
node_features.append(pixel.tolist() + pos)
current_node = i * w + j
for ni, nj in [(i-1,j),(i+1,j),(i,j-1),(i,j+1),(i-1,j-1),(i+1,j+1),(i-1,j+1),(i+1,j-1)]:
if 0 &lt;= ni &lt; h and 0 &lt;= nj &lt; w:
neighbor_node = ni * w + nj
edges.append([current_node, neighbor_node])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
x = torch.tensor(node_features, dtype=torch.float)
return Data(x=x, edge_index=edge_index)
image_noisy = image / 255.0 + np.random.normal(0, 0.08, image.shape)
image_noisy = np.clip(image_noisy, 0, 1)
noisy_np = (image_noisy * 255).astype(np.uint8)
image_median = cv2.medianBlur(noisy_np, 3)
data = image_to_graph_color_with_pos((image_noisy * 255).astype(np.uint8))
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)
target = torch.tensor(image / 255.0, dtype=torch.float).view(-1, 3).to(device)
class EnhancedColorGNN(torch.nn.Module):

def __init__(self, in_ch, out_ch):
super().__init__()
self.conv1 = GCNConv(in_ch, 64)
self.bn1 = BatchNorm1d(64)
self.conv2 = GCNConv(64, 64)
self.bn2 = BatchNorm1d(64)
self.conv3 = GCNConv(64, out_ch)
self.drop = Dropout(0.2)
def forward(self, x, edge_index):
x = torch.relu(self.bn1(self.conv1(x, edge_index)))
x = self.drop(x)
x = torch.relu(self.bn2(self.conv2(x, edge_index)))
x = self.drop(x)
x = torch.sigmoid(self.conv3(x, edge_index))
return x
model = EnhancedColorGNN(in_ch=5, out_ch=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
for epoch in range(100):
model.train()
optimizer.zero_grad()
output = model(data.x, data.edge_index)
output_img = output.view(h, w, 3).permute(2, 0, 1).unsqueeze(0)
target_img = target.view(h, w, 3).permute(2, 0, 1).unsqueeze(0)
loss = 1 - ssim(output_img, target_img, data_range=1.0, size_average=True)
loss.backward()
optimizer.step()
scheduler.step()
if (epoch + 1) % 10 == 0:
print(f&quot;Epoch {epoch+1}, SSIM Loss: {loss.item():.5f}&quot;)
output_np = output.view(h, w, 3).cpu().detach().numpy()
output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
def compute_ssim(img1, img2):
t1 = torch.tensor(img1 / 255.0, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
t2 = torch.tensor(img2 / 255.0, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
return ssim(t1, t2, data_range=1.0).item()
ssim_noisy = compute_ssim(image, noisy_np)
ssim_median = compute_ssim(image, image_median)
ssim_gnn = compute_ssim(image, output_np)
print(f&quot;SSIM - Bruitée : {ssim_noisy:.4f}&quot;)
print(f&quot;SSIM - Médian : {ssim_median:.4f}&quot;)
print(f&quot;SSIM - GNN : {ssim_gnn:.4f}&quot;)

combined = np.hstack([image, noisy_np, image_median, output_np])
titles = [&#39;Originale&#39;, &#39;Bruitée&#39;, &#39;Médian&#39;, &#39;GNN&#39;]
plt.figure(figsize=(18, 5))
for i in range(4):
plt.subplot(1, 4, i+1)
plt.imshow(combined[:, i * w:(i + 1) * w])
plt.title(titles[i])
plt.axis(&#39;off&#39;)
plt.tight_layout()
plt.show()
save_path = &quot;C:\\TraitementImage\\comparison_with_median.jpg&quot;
cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
print(f&quot;Image sauvegardée : {save_path}&quot;)