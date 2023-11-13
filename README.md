## How to use PaddleLabel for labeling table recognition 

1. Clone project PaddleLabel in github 
2. cd folder PPOCRLabel => change directory in file labelconfig.py 
  2.1. MODEL_DIR: path of text recognition model and text detection model
  2.2. EXPORT_DIR: path for saving table recognition output excel
3. Change text recognition model in table recognition in ppstructure/table/custom_predict_table.py line 72
4. Chagne model_dir for text detection in line 100 PPOCRLabel.py and text recognition in line 109

5. Run PPOCRLabel.py
