```diff
! READ FIRST! 
```
# 1. Upload ml_2dd_peach_koehler.ipynb into Google Colab and open it (you may connect to a GPU node for faster training)

# 2. In Google Colab

## 2.1. Clone the repository
>* !git config --global --unset http.proxy
>* !git config --global --unset https.proxy
>* !git config --global user.email "mrafiei1@jhu.edu"
>* !git config --global user.name "mrafiei1"
>* !git clone https://github.com/mhrafiei/ml-2dd-peach-koehler.git

## 2.2. Cat /content/ml-2dd-peach-koehler/data-matlab/data_part_* to /content/ml-2dd-peach-koehler/data-matlab/data.zip and unzip it
>* cd /content/ml-2dd-peach-koehler/data-matlab/
>* !cat data_part_a* > data.zip
>* !unzip data.zip

## 2.3. Run /content/ml-2dd-peach-koehler/code-keras/code_data.py for data processing and five RTTs and five RRSs
>* cd /content/ml-2dd-peach-koehler/code-keras/
>* !python3.6 code_data.py

## 2.4. Edit neuron architectures to be investigated in /content/ml_2dd_peach_koehler/code-keras/code_creator_keras.py and run it
>* cd /content/ml_2dd_peach_koehler/code-keras/
>* !python3.6 code_creator_keras.py


## 2.5. Edit ML input parameters in /content/ml-2dd-peach-koehler/code-keras/cls_keras.py and run any prefered /content/ml-2dd-peach-koehler/code-keras/master_keras_* file
>* #Example for RTT of 3 and RRS of 2
>* !python3.6 master_keras_3_2_100_75_50_25_12.py
 
## 2.6. Results including the models and figures are availble at /content/ml-2dd-peach-koehler/data-python/
>* cd /content/ml-2dd-peach-koehler/data-python/

## P.S. Never delete temp.log files in subdirectories 
