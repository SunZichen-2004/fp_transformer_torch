# Dialanine in water (GAFF + TIP3P, torch-sim)

经验力场全经典：溶质 GAFF（SPONGE）+ 溶剂 TIP3P，经 `MechanicalEmbeddingMLMM(enable_mace=False)`。

| 路径 | 内容 |
|------|------|
| `prepare_box.py` | 建小盒子（默认 **18 Å**，~192 水，**598 atoms**）→ `mm_box/` |
| `run_md_init.py` | 无偏 NVT 10000 step → `init_coords_{N}.npy` |
| `sFPT/sample.py` | φ/ψ sFPT（`kernel_grad_beta = λ·β0`） |
| `../dialanine_water_meta/sample.py` | φ/ψ WT-MetaD（`period=2π`） |

Slurm：

```bash
# MD 初态 → sFPT（依赖 MD）
MID=$(STEPS=10000 N_INIT=1024 sbatch --parsable scripts/dialanine_water_md_init.slurm)
REPLICAS=1024 PRE_MD_STEPS=0 sbatch --dependency=afterok:$MID scripts/dialanine_water_sfpt.slurm

# MetaD 冒烟
STEPS=200 DUMP_EVERY=50 REPLICAS=8 sbatch scripts/dialanine_water_meta.slurm
```
