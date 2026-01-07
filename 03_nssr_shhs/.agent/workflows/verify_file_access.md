---
description: Verify access to critical project files (Checkpoints, Parquet, XML)
---
This workflow verifies that the agent has access to the required data files for the NSSR SHHS Dashboard project.

1. **Verify Checkpoint Files**
   Check for `.ckpt` files in the `checkpoint_files` directory.
   ```bash
   ls -1 checkpoint_files/*.ckpt | head -n 5
   ```

2. **Verify Parquet Files**
   Check for `.parquet` files in the `parquet_files` directory.
   ```bash
   ls -1 parquet_files/*.parquet | head -n 5
   ```

3. **Verify NSRR XML Annotations**
   Check for presence of XML files in the annotation subdirectories (e.g., shhs1, shhs2).
   ```bash
   find parquet_files/annotations-events-nsrr -name "*.xml" | head -n 5
   ```

4. **Verify Profusion XML Annotations**
   Check for presence of XML files in the profusion annotation subdirectories.
   ```bash
   find parquet_files/annotations-events-profusion -name "*.xml" | head -n 5
   ```

5. **Confirm Streamlit App (Optional)**
   If the app should be running, check if it's active.
   ```bash
   lsof -i :8501
   ```
