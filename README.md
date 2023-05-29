# Maximum Mean Discrepancy-based Multi-task learning (MMD-MTL)
This repository contains codes to reproduce experiments in "Multi-task Deep Learning Methods for Improving Human Context Recognition from Low Sampling Rate Sensor Data".

We cannot release CRA and WASH datasets. To run the main MMD-MTL.py, dataloaders in data_loader.py. The dataset class should extend Dataset class as in "Dummy" or "CRA_dataset" examples.
All datasets that are considered need to be included in "task" at the beginning of MMD-MTL.py file.  