# MovieLens candidate-pool notebook

## What this notebook does
1. **Load MovieLens (1M)** – we read `ratings.csv` (or convert `ratings.dat` → `ratings.csv`).
2. **Do a time-aware leave-one-out split** – for each user:
   - last rating → **test**
   - second last → **val** (if exists)
   - the rest → **train**
   - only keep users with ≥2 ratings ≥ 4.0
   - saves to `splits/` as parquet
3. **Build ID maps** – from train only → `user_id_map.parquet`, `item_id_map.parquet`.
4. **Make a sparse matrix** (users × items) from **train** → used by ALS + item–item.
5. **Train ALS** (from `implicit`) on the train matrix.
6. **Fit item–item kNN** (from `sklearn`) on the **item × user** matrix.
7. **Build candidate pools** for val/test users by **merging**:
   - popularity (train-only)
   - ALS top-N
   - item–item neighbors
   then: dedupe, drop seen items, cap to K.
8. **Save candidates** → `candidates/val.parquet`, `candidates/test.parquet`.
9. **Compute candidate coverage** – % of users whose held-out item is inside their pool.

---

## Local Setup

### 1. Create / activate venv (Windows, VS Code)
```powershell
cd into your directory
python -m venv .venv
.\.venv\Scripts\Activate.ps1  
```

### 2. Install Python Libs
python -m pip install --upgrade pip
python -m pip install pandas numpy pyarrow scipy scikit-learn tqdm implicit
