# """
# test_tree_rl.py
# ===============
# Script test Tree-RL với 2 checkpoint.

# Cấu trúc thư mục mặc định (chạy từ "New folder"):
#     New folder/
#     ├── test_tree_rl.py
#     ├── checkpoints/
#     │   ├── epoch_001.pth
#     │   └── epoch_002.pth
#     ├── VOCdevkit/
#     │   └── VOC2007/
#     └── feature_cache/

# Cách dùng nhanh (dùng đúng default paths):
#     python test_tree_rl.py

# Cách dùng đầy đủ:
#     python test_tree_rl.py \
#         --ckpt1 checkpoints/epoch_001.pth \
#         --ckpt2 checkpoints/epoch_002.pth \
#         --voc_root VOCdevkit \
#         --feature_cache feature_cache \
#         --levels 5 \
#         --max_images 500 \
#         --output results.json
# """

# import argparse, os, sys, json
# import xml.etree.ElementTree as ET
# from collections import deque
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.ops as ops
# import torchvision.transforms as T
# from PIL import Image
# from tqdm import tqdm

# # ── CONFIG mặc định (khớp với notebook) ──────────────────────
# DEFAULT_CFG = {
#     'roi_pool_size':  7,
#     'hidden_dim':     2048,
#     'num_actions':    13,
#     'history_len':    50,
#     'scaling_ratio':  0.55,
#     'trans_ratio':    0.25,
#     'state_dim':      8842,   # 4096 + 4096 + 50*13
# }

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # ── ACTIONS ──────────────────────────────────────────────────
# def apply_action(window, action_id, sr=0.55, tr=0.25, img_w=1, img_h=1):
#     x1, y1, x2, y2 = window
#     w, h = x2 - x1, y2 - y1
#     if action_id < 5:
#         sw, sh = w * sr, h * sr
#         offsets = [(0,0),(w-sw,0),(0,h-sh),(w-sw,h-sh),((w-sw)/2,(h-sh)/2)]
#         dx, dy = offsets[action_id]
#         nx1, ny1, nx2, ny2 = x1+dx, y1+dy, x1+dx+sw, y1+dy+sh
#     else:
#         dx, dy = w * tr, h * tr
#         aid = action_id - 5
#         if   aid == 0: nx1,ny1,nx2,ny2 = x1-dx,y1,x2-dx,y2
#         elif aid == 1: nx1,ny1,nx2,ny2 = x1+dx,y1,x2+dx,y2
#         elif aid == 2: nx1,ny1,nx2,ny2 = x1,y1-dy,x2,y2-dy
#         elif aid == 3: nx1,ny1,nx2,ny2 = x1,y1+dy,x2,y2+dy
#         elif aid == 4: nx1,ny1,nx2,ny2 = x1-dx,y1,x2+dx,y2
#         elif aid == 5: nx1,ny1,nx2,ny2 = x1+dx,y1,x2-dx,y2
#         elif aid == 6: nx1,ny1,nx2,ny2 = x1,y1-dy,x2,y2+dy
#         else:          nx1,ny1,nx2,ny2 = x1,y1+dy,x2,y2-dy
#     nx1 = max(0, min(nx1, img_w-2))
#     ny1 = max(0, min(ny1, img_h-2))
#     nx2 = max(nx1+1, min(nx2, img_w))
#     ny2 = max(ny1+1, min(ny2, img_h))
#     return np.array([nx1,ny1,nx2,ny2], dtype=np.float32)

# def encode_history(history, hlen=50, nact=13):
#     enc = np.zeros(hlen * nact, dtype=np.float32)
#     for i, a in enumerate(history[-hlen:]):
#         enc[i * nact + a] = 1.0
#     return enc

# # ── METRICS ──────────────────────────────────────────────────
# def iou(a, b):
#     ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
#     ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
#     inter = max(0, ix2-ix1) * max(0, iy2-iy1)
#     union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
#     return inter / union if union > 0 else 0.0

# def recall_curve(proposals_list, gt_list, iou_thrs=(0.5, 0.6, 0.7, 0.8), size_thr=2000):
#     res = {'all': {}, 'large': {}, 'small': {}}
#     for thr in iou_thrs:
#         counts = {'all': [0,0], 'large': [0,0], 'small': [0,0]}
#         for props, gts in zip(proposals_list, gt_list):
#             for gt in gts:
#                 area = (gt[2]-gt[0]) * (gt[3]-gt[1])
#                 sk = 'large' if area > size_thr else 'small'
#                 hit = int(any(iou(p, gt) >= thr for p in props))
#                 counts['all'][0]  += hit; counts['all'][1]  += 1
#                 counts[sk][0]     += hit; counts[sk][1]     += 1
#         for k in res:
#             tot = counts[k][1]
#             res[k][thr] = counts[k][0] / tot if tot > 0 else 0.0
#     return res

# # ── DATASET ──────────────────────────────────────────────────
# VOC_CLASSES = [
#     'aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
#     'chair','cow','diningtable','dog','horse','motorbike','person',
#     'pottedplant','sheep','sofa','train','tvmonitor'
# ]
# CLS2IDX = {c: i for i, c in enumerate(VOC_CLASSES)}

# class VOCTestDataset:
#     def __init__(self, root, split='test'):
#         self.root = root
#         self.transform = T.Compose([
#             T.ToTensor(),
#             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#         ])
#         sf = os.path.join(root, 'VOC2007', 'ImageSets', 'Main', f'{split}.txt')
#         if not os.path.exists(sf):
#             raise FileNotFoundError(
#                 f'Không tìm thấy: {sf}\n'
#                 f'Hãy kiểm tra --voc_root trỏ đúng vào thư mục chứa VOC2007/'
#             )
#         with open(sf) as f:
#             self.ids = [l.strip() for l in f if l.strip()]
#         print(f'[Dataset] VOC2007 {split}: {len(self.ids)} images  ({root})')

#     def __len__(self): return len(self.ids)

#     def __getitem__(self, idx):
#         iid = self.ids[idx]
#         img_path = os.path.join(self.root, 'VOC2007', 'JPEGImages', f'{iid}.jpg')
#         img = Image.open(img_path).convert('RGB')
#         W, H = img.size
#         boxes = []
#         ann = os.path.join(self.root, 'VOC2007', 'Annotations', f'{iid}.xml')
#         for obj in ET.parse(ann).getroot().findall('object'):
#             diff = obj.find('difficult')
#             if diff is not None and int(diff.text) == 1: continue
#             cls = obj.find('name').text.strip().lower()
#             if cls not in CLS2IDX: continue
#             bb = obj.find('bndbox')
#             x1 = max(0, float(bb.find('xmin').text) - 1)
#             y1 = max(0, float(bb.find('ymin').text) - 1)
#             x2 = min(W, float(bb.find('xmax').text) - 1)
#             y2 = min(H, float(bb.find('ymax').text) - 1)
#             if x2 > x1 and y2 > y1:
#                 boxes.append([x1,y1,x2,y2])
#         gt = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), dtype=np.float32)
#         return {
#             'image':    self.transform(img),
#             'img_id':   iid,
#             'year':     '2007',
#             'img_w':    W,
#             'img_h':    H,
#             'gt_boxes': gt,
#         }

# # ── MODELS ───────────────────────────────────────────────────
# class VGG16Extractor(nn.Module):
#     def __init__(self, roi_size=7):
#         super().__init__()
#         self.roi_size = roi_size
#         vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#         self.conv = nn.Sequential(*list(vgg.features.children())[:30])
#         self.fc6  = nn.Sequential(
#             nn.Linear(512*roi_size*roi_size, 4096),
#             nn.ReLU(inplace=True), nn.Dropout(0.5)
#         )
#         if roi_size == 7:
#             self.fc6[0].weight.data = vgg.classifier[0].weight.data
#             self.fc6[0].bias.data   = vgg.classifier[0].bias.data
#         for p in self.conv.parameters(): p.requires_grad = False

#     def get_fmap(self, img):
#         with torch.no_grad(): return self.conv(img)

#     def roi_feat(self, fmap, boxes, scale=1/16):
#         bidx = torch.zeros(len(boxes), 1, dtype=boxes.dtype, device=boxes.device)
#         rois = torch.cat([bidx, boxes], 1)
#         pooled = ops.roi_pool(fmap, rois, (self.roi_size, self.roi_size), scale)
#         return self.fc6(pooled.view(len(boxes), -1))

# class QNetwork(nn.Module):
#     def __init__(self, state_dim=8842, hidden=1024, n_act=13):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, hidden), nn.ReLU(),
#             nn.Linear(hidden, hidden),   nn.ReLU(),
#             nn.Linear(hidden, hidden),   nn.ReLU(),
#             nn.Linear(hidden, n_act),
#         )
#     def forward(self, x): return self.net(x)
#     def two_best(self, s):
#         with torch.no_grad():
#             q = self.forward(s.unsqueeze(0)).squeeze(0)
#             return int(q[:5].argmax()), int(q[5:].argmax()) + 5

# # ── AGENT (inference only) ───────────────────────────────────
# class TreeRLAgent:
#     def __init__(self, extractor, q_net, cfg, device):
#         self.ext = extractor
#         self.q   = q_net
#         self.cfg = cfg
#         self.dev = device

#     def _build_state(self, gf, rf, hist):
#         return np.concatenate([
#             rf.cpu().numpy(), gf.cpu().numpy(),
#             encode_history(hist, self.cfg['history_len'], self.cfg['num_actions'])
#         ]).astype(np.float32)

#     @torch.no_grad()
#     def tree_search(self, fmap, W, H, levels=5):
#         scale = 1/16
#         gb = torch.tensor([[0,0,W,H]], dtype=torch.float32, device=self.dev)
#         gf = self.ext.roi_feat(fmap, gb, scale).squeeze(0)
#         props = []
#         queue = deque()
#         queue.append((np.array([0,0,W,H], dtype=np.float32), [], 1))
#         while queue:
#             win, hist, lvl = queue.popleft()
#             props.append(win.copy())
#             if lvl >= levels:
#                 continue
#             wb = torch.from_numpy(np.array([win], dtype=np.float32)).to(self.dev)
#             rf = self.ext.roi_feat(fmap, wb, scale).squeeze(0)
#             state = self._build_state(gf, rf, hist)
#             st = torch.tensor(state, device=self.dev)
#             bs, bt = self.q.two_best(st)
#             for a in [bs, bt]:
#                 nw = apply_action(win, a, self.cfg['scaling_ratio'],
#                                   self.cfg['trans_ratio'], W, H)
#                 queue.append((nw, hist + [a], lvl + 1))
#         return props

# # ── HELPERS ──────────────────────────────────────────────────
# def load_fmap(cache_dir, year, split, iid, device):
#     p = Path(cache_dir) / f'VOC{year}' / split / f'{iid}.pt'
#     if not p.exists():
#         return None, None, None
#     d = torch.load(p, map_location='cpu', weights_only=False)
#     fmap = d['feature_map'].float().unsqueeze(0).to(device)
#     return fmap, d['img_w'], d['img_h']

# def load_checkpoint(ckpt_path, cfg, device):
#     if not os.path.exists(ckpt_path):
#         raise FileNotFoundError(f'Không tìm thấy checkpoint: {ckpt_path}')
#     print(f'\n[Checkpoint] Loading: {ckpt_path}')
#     ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

#     extractor = VGG16Extractor(cfg['roi_pool_size']).to(device)
#     extractor.eval()

#     q_net = QNetwork(cfg['state_dim'], cfg['hidden_dim'], cfg['num_actions']).to(device)
#     q_net.load_state_dict(ckpt['q_net'])
#     q_net.eval()

#     epoch = ckpt.get('epoch', '?')
#     gstep = ckpt.get('global_step', '?')
#     print(f'  epoch={epoch}  global_step={gstep}')

#     return TreeRLAgent(extractor, q_net, cfg, device), epoch

# def run_eval(agent, dataset, feature_cache, levels, max_images, device):
#     n = len(dataset) if max_images is None else min(max_images, len(dataset))
#     all_props, all_gts = [], []

#     # Đếm cache hit/miss để thông báo
#     cache_hits = 0

#     for i in tqdm(range(n), desc=f'  Tree-search (levels={levels})', ncols=80):
#         s = dataset[i]
#         if len(s['gt_boxes']) == 0:
#             continue

#         fmap, W, H = load_fmap(feature_cache, s['year'], 'test', s['img_id'], device)
#         if fmap is not None:
#             cache_hits += 1
#         else:
#             img_t = s['image'].unsqueeze(0).to(device)
#             with torch.no_grad():
#                 fmap = agent.ext.get_fmap(img_t)
#             W, H = s['img_w'], s['img_h']

#         props = agent.tree_search(fmap, W, H, levels=levels)
#         all_props.append(props)
#         all_gts.append(s['gt_boxes'])

#     total = len(all_props)
#     print(f'  Cache: {cache_hits}/{total} hits  |  On-the-fly: {total - cache_hits}')
#     return recall_curve(all_props, all_gts), total

# def print_recall_table(res, n_images, levels, label):
#     n_props  = 2**levels - 1
#     iou_thrs = [0.5, 0.6, 0.7, 0.8]

#     # Paper reference (levels=5, 31 proposals — Table 2 NIPS 2016)
#     paper = {
#         'all':   {0.5: 68.1, 0.6: 58.7, 0.7: 43.8},
#         'large': {0.5: 78.9, 0.6: 69.8, 0.7: 53.3},
#         'small': {0.5: 23.2, 0.6: 12.5, 0.7:  4.5},
#     }

#     print(f'\n{"="*72}')
#     print(f'  {label}')
#     print(f'  Levels={levels} | Proposals/image={n_props} | Images={n_images}')
#     print(f'{"="*72}')
#     header = f'{"Category":<10}'
#     for t in iou_thrs:
#         header += f'  IoU={t:.1f}'
#     if levels == 5:
#         header += '    vs paper (0.5 / 0.6 / 0.7)'
#     print(header)
#     print(f'{"-"*72}')

#     result_dict = {}
#     for k in ['all', 'large', 'small']:
#         vals = [res[k].get(t, 0) * 100 for t in iou_thrs]
#         result_dict[k] = {str(t): round(v, 2) for t, v in zip(iou_thrs, vals)}
#         row = f'{k.capitalize():<10}'
#         for v in vals:
#             row += f'  {v:>7.1f}%'
#         if levels == 5:
#             diffs = [vals[i] - paper[k].get(t, 0) for i, t in enumerate(iou_thrs[:3])]
#             row += '    ' + ' / '.join(f'{d:+.1f}' for d in diffs)
#         print(row)
#     print(f'{"="*72}')
#     return result_dict

# # ── MAIN ─────────────────────────────────────────────────────
# def parse_args():
#     p = argparse.ArgumentParser(
#         description='Test Tree-RL với 2 checkpoints',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     p.add_argument('--ckpt1',         default='checkpoints/epoch_005.pth')
#     # p.add_argument('--ckpt2',         default='checkpoints/epoch_002.pth')
#     p.add_argument('--voc_root',      default=r'C:\Users\ACER\Downloads\New folder\VOCdevkit\VOCdevkit',
#                    help='Thư mục chứa VOC2007/')
#     p.add_argument('--feature_cache', default='feature_cache',
#                    help='Thư mục cache feature maps')
#     p.add_argument('--levels',        type=int, default=5,
#                    help='Levels tree-search  (5→31 props, 6→63 props)')
#     p.add_argument('--max_images',    type=int, default=None,
#                    help='Giới hạn số ảnh (bỏ qua = toàn bộ ~4952 ảnh)')
#     p.add_argument('--output',        default=None,
#                    help='Lưu kết quả JSON')
#     return p.parse_args()

# def main():
#     args = parse_args()
#     cfg  = DEFAULT_CFG.copy()

#     print('=' * 72)
#     print('  Tree-RL  —  Evaluation')
#     print('=' * 72)
#     print(f'Device        : {DEVICE}' +
#           (f'  ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))
#     print(f'VOC root      : {args.voc_root}')
#     print(f'Feature cache : {args.feature_cache}')
#     print(f'Levels        : {args.levels}  ({2**args.levels - 1} proposals/image)')
#     print(f'Max images    : {args.max_images or "all (~4952)"}')
#     print(f'Checkpoint 1  : {args.ckpt1}')
#     # print(f'Checkpoint 2  : {args.ckpt2}')

#     # Load dataset
#     print('\n[Dataset] Loading VOC2007 test...')
#     dataset = VOCTestDataset(args.voc_root, split='test')

#     all_results = {}
#     for label, ckpt_path in [('Checkpoint 1', args.ckpt1)]:
#     # ,
#     #                           ('Checkpoint 2', args.ckpt2)]:
#         agent, epoch = load_checkpoint(ckpt_path, cfg, DEVICE)
#         res, n_imgs  = run_eval(
#             agent, dataset,
#             feature_cache = args.feature_cache,
#             levels        = args.levels,
#             max_images    = args.max_images,
#             device        = DEVICE,
#         )
#         table = print_recall_table(res, n_imgs, args.levels,
#                                    f'{label}  (epoch {epoch})')
#         all_results[label] = {
#             'checkpoint': ckpt_path,
#             'epoch':      epoch,
#             'levels':     args.levels,
#             'n_images':   n_imgs,
#             'recall':     table,
#         }

#     # So sánh
#     print(f'\n{"="*72}')
#     print('  SO SÁNH  (All objects, IoU=0.5)')
#     print(f'{"="*72}')
#     r1 = all_results['Checkpoint 1']['recall']['all']['0.5']
#     # r2 = all_results['Checkpoint 2']['recall']['all']['0.5']
#     e1 = all_results['Checkpoint 1']['epoch']
#     e2 = all_results['Checkpoint 2']['epoch']
#     winner = 'Checkpoint 1' if r1 >= r2 else 'Checkpoint 2'
#     print(f'  Checkpoint 1 (epoch {e1:>3}) : {r1:>6.2f}%')
#     print(f'  Checkpoint 2 (epoch {e2:>3}) : {r2:>6.2f}%')
#     print(f'  → Tốt hơn : {winner}  (Δ = {abs(r1-r2):.2f}%)')
#     print(f'{"="*72}')

#     if args.output:
#         with open(args.output, 'w') as f:
#             json.dump(all_results, f, indent=2)
#         print(f'\nKết quả đã lưu: {args.output}')

# if __name__ == '__main__':
#     main()



"""
test_tree_rl_visual.py
======================
Test Tree-RL với 1 checkpoint + visualize proposals trên ảnh.

Cách dùng:
    python test_tree_rl_visual.py
    python test_tree_rl_visual.py --ckpt checkpoints/epoch_005.pth --max_images 200
"""

import argparse, os, json
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CFG = {
    'roi_pool_size' : 7,
    'hidden_dim'    : 2048,
    'num_actions'   : 13,
    'history_len'   : 50,
    'scaling_ratio' : 0.55,
    'trans_ratio'   : 0.25,
    'state_dim'     : 8842,   # 4096 + 4096 + 50*13
}

# ─────────────────────────────────────────────────────────────
# ACTIONS
# ─────────────────────────────────────────────────────────────
def apply_action(win, aid, sr=0.55, tr=0.25, W=1, H=1):
    x1, y1, x2, y2 = win
    w, h = x2 - x1, y2 - y1
    if aid < 5:
        sw, sh = w * sr, h * sr
        offs = [(0,0),(w-sw,0),(0,h-sh),(w-sw,h-sh),((w-sw)/2,(h-sh)/2)]
        dx, dy = offs[aid]
        nx1, ny1, nx2, ny2 = x1+dx, y1+dy, x1+dx+sw, y1+dy+sh
    else:
        dx, dy = w * tr, h * tr
        t = [(-dx,0,0,0),(dx,0,0,0),(0,-dy,0,0),(0,dy,0,0),
             (-dx,0,dx,0),(dx,0,-dx,0),(0,-dy,0,dy),(0,dy,0,-dy)]
        o = t[aid-5]
        nx1,ny1,nx2,ny2 = x1+o[0], y1+o[1], x2+o[2], y2+o[3]
    nx1 = max(0, min(nx1, W-2)); ny1 = max(0, min(ny1, H-2))
    nx2 = max(nx1+1, min(nx2, W)); ny2 = max(ny1+1, min(ny2, H))
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

def encode_history(hist, hlen=50, nact=13):
    enc = np.zeros(hlen * nact, dtype=np.float32)
    for i, a in enumerate(hist[-hlen:]):
        enc[i * nact + a] = 1.0
    return enc

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def iou(a, b):
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def recall_at(proposals_list, gt_list, iou_thrs=(0.5, 0.6, 0.7)):
    results = {}
    for thr in iou_thrs:
        hits = sum(
            any(any(iou(p,g) >= thr for p in props) for g in gts)
            for props, gts in zip(proposals_list, gt_list)
        )
        results[thr] = hits / len(proposals_list) if proposals_list else 0.0
    return results

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
    'chair','cow','diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor'
]

class VOCTestDataset:
    def __init__(self, root, split='test'):
        self.root = root
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        sf = os.path.join(root, 'VOC2007', 'ImageSets', 'Main', f'{split}.txt')
        if not os.path.exists(sf):
            raise FileNotFoundError(
                f'Không tìm thấy: {sf}\n'
                f'Kiểm tra --voc_root trỏ đúng thư mục chứa VOC2007/'
            )
        with open(sf) as f:
            self.ids = [l.strip() for l in f if l.strip()]
        print(f'[Dataset] VOC2007 {split}: {len(self.ids)} ảnh')

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        iid = self.ids[idx]
        img_path = os.path.join(self.root, 'VOC2007', 'JPEGImages', f'{iid}.jpg')
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        boxes, labels = [], []
        ann = os.path.join(self.root, 'VOC2007', 'Annotations', f'{iid}.xml')
        for obj in ET.parse(ann).getroot().findall('object'):
            diff = obj.find('difficult')
            if diff is not None and int(diff.text) == 1: continue
            cls = obj.find('name').text.strip().lower()
            bb = obj.find('bndbox')
            x1 = max(0,   float(bb.find('xmin').text) - 1)
            y1 = max(0,   float(bb.find('ymin').text) - 1)
            x2 = min(W,   float(bb.find('xmax').text) - 1)
            y2 = min(H,   float(bb.find('ymax').text) - 1)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)
        gt = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), dtype=np.float32)
        return {
            'image'    : self.transform(img),
            'img_id'   : iid,
            'img_path' : img_path,
            'img_w'    : W,
            'img_h'    : H,
            'gt_boxes' : gt,
            'gt_labels': labels,
        }

# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────
class VGG16Extractor(nn.Module):
    def __init__(self, roi_size=7):
        super().__init__()
        self.roi_size = roi_size
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.conv = nn.Sequential(*list(vgg.features.children())[:30])
        self.fc6  = nn.Sequential(
            nn.Linear(512*roi_size*roi_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        if roi_size == 7:
            self.fc6[0].weight.data = vgg.classifier[0].weight.data
            self.fc6[0].bias.data   = vgg.classifier[0].bias.data
        for p in self.conv.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_fmap(self, img_tensor):
        return self.conv(img_tensor)

    @torch.no_grad()
    def roi_feat(self, fmap, boxes, scale=1/16):
        bidx = torch.zeros(len(boxes), 1, dtype=boxes.dtype, device=boxes.device)
        rois = torch.cat([bidx, boxes], dim=1)
        pooled = ops.roi_pool(fmap, rois, (self.roi_size, self.roi_size), scale)
        return self.fc6(pooled.view(len(boxes), -1))


class QNetwork(nn.Module):
    def __init__(self, state_dim=8842, hidden=1024, n_act=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,    hidden), nn.ReLU(),
            nn.Linear(hidden,    hidden), nn.ReLU(),
            nn.Linear(hidden,    n_act),
        )

    def forward(self, x): return self.net(x)

    @torch.no_grad()
    def best_scale_trans(self, state_vec):
        """Trả về best scaling action (0-4) và best translation action (5-12)."""
        q = self.forward(state_vec.unsqueeze(0)).squeeze(0)
        return int(q[:5].argmax()), int(q[5:].argmax()) + 5


# ─────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────
class TreeRLAgent:
    def __init__(self, extractor, q_net, cfg, device):
        self.ext = extractor
        self.q   = q_net
        self.cfg = cfg
        self.dev = device

    def _build_state(self, gf, rf, hist):
        return np.concatenate([
            rf.cpu().numpy(),
            gf.cpu().numpy(),
            encode_history(hist, self.cfg['history_len'], self.cfg['num_actions']),
        ]).astype(np.float32)

    @torch.no_grad()
    def tree_search(self, fmap, W, H, levels=5):
        scale = 1 / 16
        gb = torch.tensor([[0, 0, W, H]], dtype=torch.float32, device=self.dev)
        gf = self.ext.roi_feat(fmap, gb, scale).squeeze(0)

        proposals = []
        queue = deque()
        queue.append((np.array([0, 0, W, H], dtype=np.float32), [], 1))

        while queue:
            win, hist, lvl = queue.popleft()
            proposals.append(win.copy())
            if lvl >= levels:
                continue
            wb = torch.from_numpy(win[None]).to(self.dev)
            rf = self.ext.roi_feat(fmap, wb, scale).squeeze(0)
            st = torch.tensor(
                self._build_state(gf, rf, hist), device=self.dev
            )
            bs, bt = self.q.best_scale_trans(st)
            for act in [bs, bt]:
                nw = apply_action(win, act,
                                  self.cfg['scaling_ratio'],
                                  self.cfg['trans_ratio'], W, H)
                queue.append((nw, hist + [act], lvl + 1))

        return proposals


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def load_fmap_cache(cache_dir, year, split, img_id, device):
    p = Path(cache_dir) / f'VOC{year}' / split / f'{img_id}.pt'
    if not p.exists():
        return None, None, None
    d = torch.load(p, map_location='cpu', weights_only=False)
    fmap = d['feature_map'].float().unsqueeze(0).to(device)
    return fmap, d.get('img_w'), d.get('img_h')


def load_checkpoint(ckpt_path, cfg, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Không tìm thấy checkpoint: {ckpt_path}')
    print(f'\n[Checkpoint] {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    extractor = VGG16Extractor(cfg['roi_pool_size']).to(device).eval()
    q_net = QNetwork(cfg['state_dim'], cfg['hidden_dim'], cfg['num_actions']).to(device)
    q_net.load_state_dict(ckpt['q_net'])
    q_net.eval()

    epoch = ckpt.get('epoch', '?')
    step  = ckpt.get('global_step', '?')
    print(f'  epoch={epoch}  global_step={step}')
    return TreeRLAgent(extractor, q_net, cfg, device), epoch


# ─────────────────────────────────────────────────────────────
# VISUALIZE
# ─────────────────────────────────────────────────────────────
def draw_proposals_on_image(img_path, proposals, gt_boxes, gt_labels,
                            iou_thresh=0.5, max_props_show=31,
                            title='', save_path=None):
    """
    Vẽ proposals (cyan) và GT boxes (màu per-class) lên ảnh gốc.
    Proposals có IoU >= iou_thresh với bất kỳ GT nào sẽ tô xanh lá.
    """
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # Màu per-class (HSV → RGB)
    cls_colors = {}
    for i, cls in enumerate(VOC_CLASSES):
        hue = int(i * 180 / len(VOC_CLASSES))
        color_bgr = cv2.cvtColor(
            np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2RGB
        )[0][0]
        cls_colors[cls] = tuple(int(c)/255 for c in color_bgr)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#1a1a2e')

    for ax_idx, ax in enumerate(axes):
        ax.imshow(img)
        ax.set_facecolor('#1a1a2e')

        if ax_idx == 0:
            # ── Trái: Tất cả proposals ──────────────────────
            shown = proposals[:max_props_show]
            for p in shown:
                # Kiểm tra hit
                hit = any(iou(p, g) >= iou_thresh for g in gt_boxes) if len(gt_boxes) else False
                color = '#00ff88' if hit else '#00cfff'
                alpha = 0.8 if hit else 0.35
                lw    = 2.0 if hit else 1.0
                rect = patches.Rectangle(
                    (p[0], p[1]), p[2]-p[0], p[3]-p[1],
                    linewidth=lw, edgecolor=color,
                    facecolor=color, alpha=alpha*0.08,
                )
                ax.add_patch(rect)
                rect2 = patches.Rectangle(
                    (p[0], p[1]), p[2]-p[0], p[3]-p[1],
                    linewidth=lw, edgecolor=color,
                    facecolor='none',
                )
                ax.add_patch(rect2)

            hits_count = sum(
                1 for p in proposals
                if len(gt_boxes) and any(iou(p,g) >= iou_thresh for g in gt_boxes)
            )
            ax.set_title(
                f'Tree-Search Proposals  ({len(shown)} shown)\n'
                f'Cyan=miss  |  Green=hit (IoU≥{iou_thresh})  |  Hits: {hits_count}/{len(proposals)}',
                color='white', fontsize=11, pad=8
            )

        else:
            # ── Phải: Best proposals + GT boxes ─────────────
            # Lấy top-K proposals theo best IoU với bất kỳ GT
            if len(gt_boxes):
                scored = [(max(iou(p,g) for g in gt_boxes), p) for p in proposals]
                scored.sort(key=lambda x: -x[0])
                top_props = scored[:10]
            else:
                top_props = [(0, p) for p in proposals[:10]]

            for score, p in top_props:
                color = '#00ff88' if score >= iou_thresh else '#ffaa00'
                lw = 2.5 if score >= iou_thresh else 1.5
                rect = patches.Rectangle(
                    (p[0], p[1]), p[2]-p[0], p[3]-p[1],
                    linewidth=lw, edgecolor=color, facecolor='none',
                    linestyle='--' if score < iou_thresh else '-',
                )
                ax.add_patch(rect)
                ax.text(p[0]+2, p[1]+12, f'{score:.2f}',
                        color=color, fontsize=7.5, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))

            # GT boxes
            for g, lbl in zip(gt_boxes, gt_labels):
                color = cls_colors.get(lbl, (1,1,0))
                rect = patches.Rectangle(
                    (g[0], g[1]), g[2]-g[0], g[3]-g[1],
                    linewidth=3, edgecolor=color, facecolor=color, alpha=0.12,
                )
                ax.add_patch(rect)
                rect2 = patches.Rectangle(
                    (g[0], g[1]), g[2]-g[0], g[3]-g[1],
                    linewidth=3, edgecolor=color, facecolor='none',
                )
                ax.add_patch(rect2)
                ax.text(g[0]+2, g[1]-6, lbl,
                        color='white', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor=color, alpha=0.85))

            best_iou_val = max(
                (max(iou(p,g) for g in gt_boxes) for p in proposals),
                default=0.0
            ) if len(gt_boxes) else 0.0
            ax.set_title(
                f'Top-10 Proposals + GT Boxes\n'
                f'Best IoU = {best_iou_val:.3f}  |  '
                f'Classes: {", ".join(set(gt_labels)) if gt_labels else "none"}',
                color='white', fontsize=11, pad=8
            )

        for sp in ax.spines.values():
            sp.set_edgecolor('#444')
        ax.tick_params(colors='#888')

    fig.suptitle(title, color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'  Saved → {save_path}')
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# EVAL + VISUALIZE
# ─────────────────────────────────────────────────────────────
def run_eval_and_visualize(agent, dataset, args, epoch):
    n = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))
    all_props, all_gts = [], []
    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    vis_indices = set()  # Chọn ảnh để visualize
    vis_every   = max(1, n // args.vis_count)

    print(f'\n[Eval] {n} ảnh | levels={args.levels} '
          f'({2**args.levels - 1} proposals/img)')
    print(f'[Vis]  Lưu {args.vis_count} ảnh vào: {vis_dir}/')

    for i in tqdm(range(n), desc='Tree-search', ncols=80):
        s = dataset[i]
        if len(s['gt_boxes']) == 0:
            continue

        # Load feature map
        year = s.get('year', '2007')
        fmap, fw, fh = load_fmap_cache(
            args.feature_cache, year, args.split, s['img_id'], DEVICE
        )
        if fmap is None:
            img_t = s['image'].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                fmap = agent.ext.get_fmap(img_t)
            fw, fh = s['img_w'], s['img_h']

        proposals = agent.tree_search(fmap, fw or s['img_w'], fh or s['img_h'],
                                      levels=args.levels)
        all_props.append(proposals)
        all_gts.append(s['gt_boxes'])

        # Visualize mỗi vis_every ảnh
        do_vis = (i % vis_every == 0 and len(vis_indices) < args.vis_count)
        # Thêm: ảnh có best IoU cao / thấp để phong phú
        if do_vis or (len(vis_indices) < args.vis_count and
                      len(all_props) % (vis_every*2) == 0):
            vis_indices.add(i)
            best = max(
                (max(iou(p,g) for g in s['gt_boxes']) for p in proposals),
                default=0.0
            )
            save_p = vis_dir / f"img_{i:04d}_{s['img_id']}_iou{best:.2f}.png"
            draw_proposals_on_image(
                img_path   = s['img_path'],
                proposals  = proposals,
                gt_boxes   = s['gt_boxes'],
                gt_labels  = s['gt_labels'],
                iou_thresh = args.iou_thresh,
                title      = (f"[epoch {epoch}] {s['img_id']}  |  "
                              f"levels={args.levels}  best_IoU={best:.3f}"),
                save_path  = str(save_p),
            )

    return all_props, all_gts


def print_results(recall, n_images, levels, epoch):
    n_props  = 2**levels - 1
    thrs = [0.5, 0.6, 0.7]
    # Paper ref (NIPS 2016 Table 2, levels=5)
    paper = {0.5: 68.1, 0.6: 58.7, 0.7: 43.8}

    print(f'\n{"="*60}')
    print(f'  Kết quả — epoch {epoch}')
    print(f'  Levels={levels} | {n_props} proposals/image | {n_images} ảnh')
    print(f'{"="*60}')
    print(f'{"IoU thr":<12}{"Recall":>10}{"vs Paper":>14}')
    print(f'{"-"*40}')
    for t in thrs:
        r  = recall.get(t, 0) * 100
        diff = f'{r - paper[t]:+.1f}' if t in paper else 'N/A'
        bar_len = int(r / 2)
        bar = '█' * bar_len + '░' * (50-bar_len)
        print(f'  IoU≥{t:.1f}   {r:>6.2f}%   {diff:>8}  [{bar[:30]}]')
    print(f'{"="*60}')
    print(f'  Paper (NIPS 2016): 68.1% / 58.7% / 43.8% @ IoU 0.5/0.6/0.7')
    print(f'{"="*60}')


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Test Tree-RL + Visualize proposals',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--ckpt',
                   default='checkpoints/epoch_005.pth')
    p.add_argument('--voc_root',
                   default=r'C:\Users\ACER\Downloads\New folder\VOCdevkit\VOCdevkit',
                   help='Thư mục chứa VOC2007/')
    p.add_argument('--feature_cache',
                   default='feature_cache')
    p.add_argument('--split',
                   default='test',
                   help='trainval hoặc test')
    p.add_argument('--levels',
                   type=int, default=5,
                   help='Độ sâu tree (5→31 proposals, 6→63)')
    p.add_argument('--iou_thresh',
                   type=float, default=0.5)
    p.add_argument('--max_images',
                   type=int, default=None,
                   help='None = toàn bộ ~4952 ảnh')
    p.add_argument('--vis_count',
                   type=int, default=20,
                   help='Số ảnh visualize và lưu ra')
    p.add_argument('--vis_dir',
                   default='vis_results',
                   help='Thư mục lưu ảnh visualization')
    p.add_argument('--output',
                   default=None,
                   help='Lưu kết quả JSON (tuỳ chọn)')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = DEFAULT_CFG.copy()

    print('=' * 60)
    print('  Tree-RL  —  Evaluation + Visualization')
    print('=' * 60)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f'Device       : {DEVICE}  ({gpu_name})')
    print(f'Checkpoint   : {args.ckpt}')
    print(f'VOC root     : {args.voc_root}')
    print(f'Split        : {args.split}')
    print(f'Levels       : {args.levels}  ({2**args.levels-1} proposals/img)')
    print(f'Visualize    : {args.vis_count} ảnh → {args.vis_dir}/')

    # Load dataset + checkpoint
    dataset = VOCTestDataset(args.voc_root, split=args.split)
    agent, epoch = load_checkpoint(args.ckpt, cfg, DEVICE)

    # Eval + visualize
    all_props, all_gts = run_eval_and_visualize(agent, dataset, args, epoch)

    # Recall
    recall = recall_at(all_props, all_gts, iou_thrs=(0.5, 0.6, 0.7))
    print_results(recall, len(all_props), args.levels, epoch)

    # Lưu kết quả JSON nếu cần
    if args.output:
        result = {
            'checkpoint' : args.ckpt,
            'epoch'      : epoch,
            'levels'     : args.levels,
            'n_images'   : len(all_props),
            'recall'     : {str(k): round(v*100, 2) for k,v in recall.items()},
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'\nKết quả đã lưu: {args.output}')

    print(f'\nVisualization đã lưu tại: {args.vis_dir}/')


if __name__ == '__main__':
    main()