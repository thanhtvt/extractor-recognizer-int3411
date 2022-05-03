# Feature Extractor and Recognizer  
This is the assignment for problem 2 of "Speech processing" - INT3411 class. The program can be divided into 2 seperate parts:  
- Extract MFCCs  
- Recognize a word using GMM-HMM or DTW algorithm  
  
## Installation  
Once you have created your Python environment (recommend using conda or virtualenv), simply type:  

```bash
git clone https://github.com/thanhtvt/extractor-recognizer-int3411.git

cd extractor-recognizer-int3411

pip install -r requirements.txt
```  
  
## Usage  
There are 2 different parts of the program.  
  
Moving to `int3411` directory and then:  
  
### MFCC Extractor
Run:  
```bash
python3 feature_extractor.py [-sf] [-sm] audio_path label_path
```  
Where:  
- `audio_path`: Path to audio file  
- `label_path`: Path to corresponding label file  
- `-sf`: Whether to save figure of MFCCs
- `-sm`: Whether to save mat file of MFCCs  
  
### Recognizer  
Run:
```bash
python3 recognizer.py [-i INPUT_MAT] -d MAT_DIR [-r {dtw,gmm-hmm] [-m PRETRAINED_MODEL]
```  
Where  
- `-i`: Path to mat file to predict label (Use all mat files if not specified)
- `-d`: Path to mat dir
- `-r`: Type of model used (`dtw` or `gmm-hmm`)
- `-m`: Path to GMM-HMM pretrained model
  
## Demo  
  