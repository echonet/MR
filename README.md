# EchoNet MR

# Running Inference

Run the following in the command line, specifying arguments: 

python predict.py --MR_label_col <MR_label_col> --manifest_path <manifest_path> --data_path <data_path>

--MR_label_col: specify column where your ground truth labels are present </br>
--manifest_path: Path to your manifest </br>
--data_path: Path to AVIs </br>

ex: python predict.py --MR_label_col MR_severity --manifest_path C:\Users\Remote\Documents\Amey\MR\manifest_for_testing_ext_val_repo.csv --data_path D:\amey\stanford_echos_MR_ext_val

<img width="1083" alt="image" src="https://github.com/ecg-net/lvef/blob/main/EF_screenshot.png">


### Obtaining Stats and Figures</br>

After running predict.py, run all cells in the notebook "analyze_predictions.ipynb"

<img width="1083" alt="image" src="https://github.com/echonet/MR/assets/111397367/03b51ec7-f062-4c7d-8657-3be85d49128c">

</br></br> </br>

![](https://github.com/ecg-net/lvef/blob/main/analyze_predictions_lvef_repo.gif)


