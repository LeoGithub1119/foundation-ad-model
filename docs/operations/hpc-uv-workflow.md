# HPC + uv 工作流程

本頁記錄本專案可重複使用的 HPC 與 uv 環境管理方式。

## 基本模式

每個研究 repo 都複製一份自己的 `hpc/` toolkit。穩定邏輯放在 shell scripts，專案路徑與實驗設定放在 config files。

```text
repo/
  hpc/
    bin/
      activate.sh
      download_hf.sh
      run_train.sh
      run_eval.sh
    slurm/
    templates/
    project.env      # local, ignored
    project.config   # local, ignored
```

## 分層原則

| 層級 | 檔案 / 目錄 | 用途 | Git policy |
| --- | --- | --- | --- |
| 可重用 HPC toolkit | `hpc/bin/`, `hpc/slurm/`, `hpc/templates/` | 從 `hpc-tools` 複製後依 repo 做最小調整 | tracked |
| 專案環境 | `hpc/project.env` | work dir、venv path、module names | ignored |
| 實驗設定 | `hpc/project.config` | model ID、dataset ID、experiment name | ignored，直到整理出可分享 example |
| Python 環境 | `.venv310_dino/` | repo-local uv-managed venv | ignored |
| Runtime artifacts | datasets、models、outputs、logs、checkpoints | 產生或下載的資料 | ignored |

## uv 使用規則

使用 `uv` 建立或更新 repo-local environment。不要 commit virtual environments 或下載套件。

建立環境範例：

```bash
uv venv .venv310_dino --python 3.10
uv pip install --python .venv310_dino/bin/python -r requirements.txt
```

預覽文件網站：

```bash
uv pip install --python .venv310_dino/bin/python -r requirements-docs.txt
.venv310_dino/bin/mkdocs serve
```

安裝實驗依賴：

```bash
uv pip install --python .venv310_dino/bin/python -r requirements-experiments.txt
```

## 互動式 GPU Smoke Test

短 smoke test 建議在 compute node 上跑，不要在 login node 上跑。

常用 allocation 指令：

```bash
salloc -p normal -N 1 --gres=gpu:1 --cpus-per-task=4 -t 00:20:00 --account=<ACCOUNT_ID>
srun --pty bash
```

or:

```bash
srun --account=<ACCOUNT_ID> -p normal --gres=gpu:1 -c 8 -t 0:20:00 --pty bash
```

進入節點後：

```bash
cd ~/DINO
bash hpc/bin/run_smoke.sh
```

正式實驗可依預期 runtime 與 debug 需求選擇 `sbatch`，或先 allocation node、開 `tmux`，再用 `srun` 啟動。

第一個 supervised baseline：

```bash
bash hpc/bin/run_train.sh
```

## 目前 DINO 環境

| 項目 | 目前值 |
| --- | --- |
| Work directory | `WORK_DIR`，由本機 `hpc/project.env` 指定 |
| Local venv | `.venv310_dino/` |
| 第一個 model target | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| 第一階段實驗類型 | frozen encoder supervised anomaly detection |

## 可重現性注意事項

- `hpc/project.env` 包含 machine-specific paths，保持 local、不進 git。
- `hpc/project.config` 在實驗仍快速變動時保持 local。
- 當某組設定變成可報告實驗後，再整理成 tracked example file。
- 每個可報告實驗都要登記到 [實驗總表](../experiments/registry.md)。
