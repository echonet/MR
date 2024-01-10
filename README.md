# EchoNet MR

# Running Inference

Run the following in the command line, specifying arguments: 

python predict.py --MR_label_col <MR_label_col> --manifest_path <manifest_path> --data_path <data_path>

--MR_label_col: specify column where your ground truth labels are present </br>
      
      MR severity for each video should be one of the following: "Control", "Mild", "Moderate", "Severe". 
      
      When MR is mild to moderate, please round up to "Moderate".
      
      If MR is moderate to severe, then please round up to "Severe". </br>
      

--manifest_path: Path to your manifest </br>
--data_path: Path to AVIs </br>

ex: python predict.py --MR_label_col MR_severity --manifest_path C:\Users\Remote\Documents\Amey\MR\manifest_for_testing_ext_val_repo.csv --data_path D:\amey\stanford_echos_MR_ext_val

### Obtaining Stats and Figures</br>

After running predict.py, run all cells in the notebook "analyze_predictions.ipynb"
