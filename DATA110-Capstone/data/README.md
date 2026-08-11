# Data Setup

The project uses NASA C-MAPSS **FD004**. The raw benchmark is not committed to GitHub so that the repository remains small and avoids redistributing the source archive.

## Download

Download `CMAPSSData.zip` from the NASA Open Data Portal dataset page or the resource URL shown in the project report.

Extract these files into this directory:

```text
data/
└── raw/
    ├── train_FD004.txt
    ├── test_FD004.txt
    └── RUL_FD004.txt
```

The loader also accepts the standard NASA directory layout if the files are placed directly in `data/raw/`.

## Expected Columns

Each trajectory row contains:
- unit number
- time/cycle
- three operational settings
- 21 sensor measurements

The official test RUL file provides the remaining useful life for the final observed cycle of each test engine.

## Reproducibility Note

Do not edit the raw files. All transformations should be performed by the scripts in `src/` so that the preprocessing and feature engineering steps remain auditable.
