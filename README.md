# L3-apnea-AI.github.io
A Sleep Apnea Prediction tool for hospitals that employs a 1D CNN on features extracted from ECG signals obtained from a single sensor to monitor sleep apnea disease at home. The tool utilizes data obtained from Level 3 vital sign monitoring, and is designed to accurately predict the likelihood of sleep apnea in patients. 
This application predicts wheather the patient is suffering from Apnea or not. The application is divided into three sections: data exploration, data visualization, and prediction.


### Programming Platforms Used
* Python 3
* MatplotLib
* NumPy
* Pandas
* Keras
* Seaborn
* scikit-learn
* Streamlit




### Input to 1D CNN
The raw files taken from Apnea-ECG Database PhysioNet consist of 70 records, divided into a learning set of 35 records (a01 through a20, b01 through b05, and c01 through c10), and a test set of 35 records (x01 through x35), all of which may be downloaded from the zip file apnea-ecg-database-1.0.0.zip in the below link.
* https://drive.google.com/file/d/1C-4Lu7l4rNwHMGQLqnV0vYkkNeUGgL3T/view?usp=sharing


This zip file contains .dat, .apn, .hea, .qrs, and other files. From this data, 70 Patient records were created, each of which contained a patient's minute-minute ecg signals and their corresponding annotations (Apnea(A) or Nonapnea(N). These 35 training datasets were combined into a single csv file , trained with a one-dimensional neural network, and saved to an h5 file which is given in the below link


* Single file consisting of 35 patient records with annotations: https://drive.google.com/file/d/1vIn_bFy7RmMbuSFIkDttiPKK9ph8_MQu/view?usp=sharing
* Trained 1D CNN model: https://drive.google.com/file/d/1shoCvp_k-M3-8fFh1MyuaQ0bU2GLp3Xp/view?usp=sharing
### Input to Streamlit
Sequence of hourly measurements of ecg signals of a patient with single column consisting of one channel as in the following link: https://github.com/kxrtxkx/L3-apnea-AI/commit/bd0c23df45a919a0d42921f6ff63f43990d9ed06


### Usage
To run the app, run `streamlit run app.py` in your terminal.
AI Sleep Apnea scoring tool has 3 sections:
1) Data Exploration
 * Upload the file in AI sleep apnea tool. The nan values are filled using forward fill.
 The user is allowed to give a sampling rate in hertz, inorder to resample hourly data into minute-minute data.
2) Data Visualization
* Displays a visualisation of each row 
3) Prediction
* This section allows users to upload the trained h5 model and predict whether the patient suffers from apnea or not. The apnea-hypopnea index (AHI) is the combined average number of apneas and hypopneas that occur per hour of sleep, and the average apneas per hour and the severity of apnea are given as results. Additionally, a heat map of the number of apneas in each hour is also provided.
