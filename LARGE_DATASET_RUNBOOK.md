# Running ImageNet-LT and iNaturalist-2018

Step-by-step procedure for the two large benchmarks on a Linux workstation. The CIFAR
benchmarks need none of this — they download themselves.

Everything below assumes the repo is at `~/CSL-PF/CSL-Extended-clean`; substitute your own path.

---

## 0. Sync the code

On the machine you edit from:

```bash
git add -A && git commit -m "wire the large long-tailed benchmarks" && git push
```

On the workstation:

```bash
cd ~/CSL-PF/CSL-Extended-clean
git pull
source .venv/bin/activate          # not conda base
python run_table.py --check
```

Both CIFAR rows should read READY and both large rows NOT AVAILABLE, listing the environment
variable each one wants. If `--check` fails to run at all, stop here and fix that first — it
imports nothing heavier than a few dictionaries.

## 1. Confirm the machine can hold it

```bash
df -h ~
nvidia-smi --query-gpu=name,memory.total --format=csv
nproc
```

**Disk.** Do one dataset at a time and delete each archive immediately after extracting, or you
will need double the space.

| | Archive | Extracted | Peak during setup |
|---|---|---|---|
| ImageNet-LT | 145 GB | ~145 GB | ~290 GB |
| iNaturalist-2018 | 120 GB | ~120 GB | ~240 GB |
| Both, archives deleted | — | ~265 GB | — |

**GPU.** The paper used a single Tesla V100 32GB at batch 256. With less than about 24GB, add
`--batch 128 --lr-scale 0.5` to the run commands; linear scaling keeps that an honest
approximation, and it should be reported as a deviation.

**Workers.** The default is 8 dataloader workers. At 224px, JPEG decoding is usually the
bottleneck rather than the GPU, so if `nproc` gives you plenty, more workers help.

## 2. Download iNaturalist-2018 first

Public, so it needs no account and can run unattended while you deal with ImageNet
registration. Do it in tmux — this takes hours.

Download *outside* the repo. `/data` needs root on most machines, so use your home directory —
and note that starting `wget` from wherever your shell happens to sit will drop 120GB into the
repository, which is a nuisance to undo afterwards.

```bash
mkdir -p "$HOME/data/inat" && cd "$HOME/data/inat"
tmux new -s dl-inat
wget -c https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz
```

`-c` resumes a partial download, so a dropped connection is not fatal. Ctrl+B then D to detach.

When it finishes, extract inside tmux as well — unpacking 120GB of gzip takes hours:

```bash
cd "$HOME/data/inat"
tar -xzf train_val2018.tar.gz      # gzip verifies checksums, so corruption fails loudly here
ls train_val2018                   # expect Actinopterygii, Amphibia, Animalia, Aves, ...
du -sh train_val2018
rm train_val2018.tar.gz            # only after you confirm the extraction worked
```

**The root path is the parent.** The split files already begin with `train_val2018/`, so:

```bash
export INATURALIST_ROOT="$HOME/data/inat"     # correct — contains train_val2018/
# NOT $HOME/data/inat/train_val2018
```

## 3. Download ImageNet

Register at <https://image-net.org>, then fetch `ILSVRC2012_img_train.tar` (138 GB) and
`ILSVRC2012_img_val.tar` (6.3 GB) — again inside tmux.

Both need rearranging into the layout the split files expect.

**Training images.** The outer tar contains 1000 inner tars, one per class, each of which has
to be unpacked into its own directory:

```bash
mkdir -p "$HOME/data/imagenet/train" && cd "$HOME/data/imagenet/train"
tar -xf /path/to/ILSVRC2012_img_train.tar
for f in *.tar; do
  d="${f%.tar}"; mkdir -p "$d"; tar -xf "$f" -C "$d"; rm "$f"
done
ls | wc -l                          # expect 1000
```

**Validation images.** These arrive as 50,000 loose JPEGs and need sorting into per-class
directories:

```bash
mkdir -p "$HOME/data/imagenet/val" && cd "$HOME/data/imagenet/val"
tar -xf /path/to/ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
ls | wc -l                          # expect 1000, not 50000
```

Then point at the parent holding both:

```bash
export IMAGENET_LT_ROOT="$HOME/data/imagenet"    # contains train/ and val/
```

## 4. Make the paths stick

Environment variables set in one shell do not survive a new tmux window, and a sweep that
starts and then cannot find its data has wasted your queue slot:

```bash
cat >> ~/.bashrc <<'EOF'
export IMAGENET_LT_ROOT="$HOME/data/imagenet"
export INATURALIST_ROOT="$HOME/data/inat"
EOF
source ~/.bashrc
```

## 5. Verify before committing days of compute

```bash
cd ~/CSL-PF/CSL-Extended-clean && source .venv/bin/activate
python run_table.py --check
```

Both large rows should now read READY, each showing which split it will evaluate on —
`test` for ImageNet-LT, `val` for iNaturalist-2018. These are not interchangeable; they are
the sets the published accuracies are measured against.

If the layout is wrong, the loader says so immediately and names the prefix it expected,
rather than failing on the first batch after the model and diffusion pipeline are built.

Then a short end-to-end run per dataset, to prove the plumbing:

```bash
python run_table.py --dataset imagenet_lt --seeds 1 --rounds 1 --no-stable-diffusion
python run_table.py --dataset inaturalist --seeds 1 --rounds 1 --no-stable-diffusion
```

These still train for the full epoch count, so interrupt them once you have seen the per-class
counts and a few epochs of progress. What you are checking:

- ImageNet-LT reports 115,846 train images over 1000 classes, 1280 down to 5 per class.
- iNaturalist reports 437,513 train images over 8142 classes, 1000 down to 2 per class.
- The GPU is busy in `nvidia-smi`.

## 6. The real runs

One seed each. Five-seed statistics are out of reach at this scale; report these as single runs.

```bash
tmux new -s imagenet
cd ~/CSL-PF/CSL-Extended-clean && source .venv/bin/activate
python run_table.py --dataset imagenet_lt --seeds 1
```

```bash
tmux new -s inat
cd ~/CSL-PF/CSL-Extended-clean && source .venv/bin/activate
python run_table.py --dataset inaturalist --seeds 1 --prompts 2 --images 1
```

Run them one at a time unless the machine has two free GPUs; pass `--gpu 1` to the second if it
does.

**Why iNaturalist needs `--prompts 2 --images 1`.** Synthesis runs over the tail, which is the
bottom 30% of classes — 2,442 of them here. At the default 50 prompts and 4 images each that is
488,400 generated images per round, over a week of diffusion alone. Two prompts and one image
each gives 4,884 per round, roughly two hours. If even that is too much, `--no-stable-diffusion`
keeps the feature-space path; just report which you used.

## 7. Monitoring

The runner prints a progress line every two minutes. For the full stream:

```bash
tail -f table_results/imagenet_lt_natural-lt_*/seed0.log
tmux attach -t imagenet
```

Early in the log, confirm the Stable Diffusion line reads `keeping pipeline resident`. If it
says `streaming weights` on a large card, restart with `--no-sd-low-vram` — the difference is
two to three times the generation time.

## 8. Results

Each sweep writes to `table_results/<dataset>_<setting>_<timestamp>/`:

- `table.md` — the comparison table, with the paper's row already filled in
- `table.json` — full statistics, per-seed results and the resolved protocol
- `seed0.log` — the complete run log
- `seed0/final_report_*.json` — per-class accuracies and per-round progression

## Reference targets

| Dataset | CE baseline | LDAL best | LDAL 5-seed |
|---|---|---|---|
| CIFAR-10-LT (p=100) | 70.40 | 80.19 | 79.77 ± 0.73 |
| CIFAR-100-LT (p=100) | 38.32 | 49.79 | 48.88 ± 0.86 |
| ImageNet-LT | 38.88 | 50.10 | 49.67 ± 0.66 |
| iNaturalist-2018 | 57.30 | 67.10 | 66.53 ± 0.83 |

## Protocol

| | Backbone | Epochs | LR | Decay | Weight decay | Batch | Eval split |
|---|---|---|---|---|---|---|---|
| CIFAR-10/100-LT | ResNet-32 | 200 | 0.1 | ×0.01 @ 160/180 | 2e-4 | 128 | test |
| ImageNet-LT | ResNet-50 | 120 | 0.1 | ×0.1 @ 60/80 | 2e-4 | 256 | test |
| iNaturalist-2018 | ResNet-50 | 160 | 0.05 | ×0.1 @ 120/145 | 1e-4 | 256 | val |

The paper states the backbone and epoch counts for the large datasets, and the full CIFAR
schedule, directly. It does not print LR schedules for ImageNet-LT or iNaturalist, saying only
that hyperparameters were adopted from AREA (Chen et al., ICCV 2023) — so the two large-dataset
schedules above follow the standard long-tailed convention and should be attributed that way
rather than to the paper.

## Time and troubleshooting

Rough single-GPU expectations: CIFAR about 1.5–4 hours per seed, ImageNet-LT one to two days,
iNaturalist-2018 closer to a week.

| Symptom | Cause |
|---|---|
| `IMAGENET_LT_ROOT points at ... not a directory` | Typo, or the variable was not exported in this shell |
| `N of the first 32 images were not found` | Root is one level too deep or too shallow; the message names the expected prefix |
| `val | wc -l` gives 50000 | `valprep.sh` was not run |
| CUDA out of memory | Add `--batch 128 --lr-scale 0.5` |
| Terminal looks frozen | Normal; progress is in `seed0.log`, with a heartbeat every two minutes |
| Sweep dies when SSH drops | It was not started inside tmux |
