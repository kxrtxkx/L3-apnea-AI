import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import joblib as jb
import mysql.connector
from streamlit_option_menu import option_menu
from configparser import ConfigParser
from torch import _fake_quantize_per_tensor_affine_cachemask_tensor_qparams
CNX: mysql.connector.connect
from sklearn.model_selection import train_test_split
import plotly.express as px
import joblib
import login
from auth import *
import pandas as pd
import requests
import h5py
import  shutil
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sn
from numpy import mean
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf
import keras
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from io import BytesIO
import tempfile
import scipy.interpolate
import scipy.signal
from scipy import signal
import pandas as pd
import streamlit as st
from scipy.signal import resample_poly
from scipy.signal import resample
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
from scipy.interpolate import interp1d

def prediction_table(avrg):
    if 0 <= avrg <= 5:
        return "No apnea"
    elif 5 < avrg <= 15:
        return "Mild apnea"
    elif 15 < avrg <= 30:
        return "Moderate apnea"
    elif avrg > 30:
        return "Severe apnea"
    return result


def column_to_rows(data, sampling_rate):
    num_rows = len(data) // (sampling_rate * 60)
    new_data = []
    for i in range(num_rows):
        start = i * sampling_rate * 60
        end = start + sampling_rate * 60
        row = data.iloc[start:end]

        if row.shape[0] != sampling_rate * 60:  # Check if the row length is correct
            continue

        # Resample row using Fourier resampling
        if sampling_rate > 100:
            n_samples = 100 * 60
            t = np.arange(row.shape[0]) / sampling_rate
            t_resampled = np.arange(n_samples) / 100
            f = interp1d(t, row, kind='linear', axis=0)
            row = f(t_resampled)

        new_data.append(row)

    new_data = np.array(new_data)
    new_data = np.squeeze(new_data, axis=2)  # remove extra axis
    return pd.DataFrame(new_data)



def sleep():
    st.write("""<style>.title {background-color: #3c3f41;color: white;padding: 10px;font-size: 40px;text-align: center;}
    </style><div class="title">AI SLEEP APNEA SCORING SYSTEM</div>
    """, unsafe_allow_html=True)
    st.subheader(""" This application predicts the sleep apnea score. The app has 3 section : """)
    st.write("- Data Exploration")
    st.write("- Data Visualization")
    st.write("- AHI Prediction")

    tab1, tab2, tab3 = st.tabs(["🏠 Data Exploration", "📈 Data Visualization", " 🔮 Prediction"])
    with tab1:
        st.write("""
        <style>
        .subtitle {
        background-color: #3c3f41;
        color: white;
        padding: 10px;
        font-size: 25px;
        text-align: center;
        }
        </style><div class="subtitle">Data Exploration </div>
        """, unsafe_allow_html=True)

        data_type = st.selectbox("Select data type", ["ECG", "HRV"])
        data = None  # Move the definition outside the if-else block
        if data_type == "ECG":
            st.markdown("<h2>Upload an ECG file</h2>", unsafe_allow_html=True)
            st.warning("(The ECG file uploading must be a single column consisting of one channel.)", icon="⚠️")
            uploaded_file_0 = st.file_uploader("", key='uploaded_file_0')
            df = pd.DataFrame()
            new_data = pd.DataFrame()
            if uploaded_file_0:
                data = pd.read_csv(uploaded_file_0)
                st.write("Raw data from uploaded file:")
                st.write(data)
                st.info("(Your data may have nan values,it is filled with forward fill. )")
                st.write(f"NAN imputation using forward fill")
                data.replace([0, 0.0, 0.00, 0.000], method='ffill', inplace=True)
                data.fillna(method='ffill', inplace=True)
                st.write(data)

                sampling_rate_input = st.text_input("Enter Sampling Rate of ECG signal in hertz:")
                try:
                    sampling_rate = int(sampling_rate_input)
                except ValueError:
                    st.warning("Enter the sample rate greater than or equal to 100", icon="⚠️")

                p = sampling_rate * 60
                st.write(f"No of samples per minute : {sampling_rate}*60 =", p)
                st.info("We have resampled it to 100 hertz :")
                sampling_rate = int(sampling_rate_input)
                new_data = column_to_rows(data, sampling_rate)
                st.write(new_data)
        else:
            st.markdown("<h2>Upload an HRV file</h2>", unsafe_allow_html=True)
            st.warning("(The HRV file uploading must be a single column consisting of one channel.)", icon="⚠️")
            uploaded_file_0 = st.file_uploader("", key='uploaded_file_0')
            df = pd.DataFrame()
            new_data = pd.DataFrame()
            if uploaded_file_0:
                data = pd.read_csv(uploaded_file_0)
                st.write("Raw data from uploaded file:")
                st.write(data)

        with tab2:
            st.write("""
            <style>
            .subtitle {
            background-color: #3c3f41;
            color: white;
            padding: 10px;
            font-size: 20px;
            text-align: center;
            }
            </style><div class="subtitle">Data Visualisation </div>
            """, unsafe_allow_html=True)

            if data_type == "ECG" and new_data is not None:
                st.header("Row-wise Visualizing for ECG")
                indices = range(len(new_data))
                selected_index = st.selectbox('Select a row to be visualised:', options=indices)
                if selected_index is not None:
                    selected_row = new_data.iloc[selected_index].to_frame().T
                    st.write("Selected Patient row", selected_row)
                    selected_row.drop(new_data.columns[[0, 1]], axis=1, inplace=True)
                    df_row = pd.DataFrame(selected_row).T
                    st.write("Selected Row-Visualisation")
                    st.write(df_row.shape)
                    st.line_chart(df_row)
            elif data_type == "HRV" and data is not None:
                st.header("Row-wise Visualizing for HRV")
                indices = range(len(data))
                selected_index = st.selectbox('Select a row to be visualised:', options=indices)
                if selected_index is not None:
                    selected_row = data.iloc[selected_index].to_frame().T
                    st.write("Selected Patient row", selected_row)
                    df_row = pd.DataFrame(selected_row).T
                    st.write("Selected Row-Visualisation")
                    st.write(df_row.shape)
                    st.line_chart(df_row)
            else:
                st.warning("Please upload a valid ECG/HRV file and perform Data Exploration to visualize the data.",
                           icon="⚠️")

        #else:
            #st.warning("No row selected.")
    with tab3:
        st.write("""
            <style>
            .subtitle {
            background-color: #3c3f41;
            color: white;
            padding: 10px;
            font-size: 20px;
            text-align: center;
            }
            </style><div class="subtitle">ML Model Prediction </div>
            """, unsafe_allow_html=True)

        st.header("SleepApnea Score & AHI prediction")
        st.subheader("1D CNN")
        st.info("(We used 1 Dimensional Convolutional Neural Network AI Model for AHI scoring. )")
        #option = st.selectbox('Choose:', ('1D CNN','AHI'), key="unique_col3")
            # uploaded_file_ann = st.file_uploader("Choose the h5 file to be loaded for testing", type="h5")
        #uploaded_file_cnn= "C:\\Users\\user\\Downloads\\newmodel.h5"
        converted_predictions = None
        st.markdown('Choose the existing trained model for sleep apnea prediction (in h5 format).')
        st.write(
            "If there is no trained model you can use the model given for  [ECG](https://drive.google.com/file/d/1O31WVMMPYlsOuE0cTg_O_BEGW-S-2Pke/view?usp=sharing) or  [HRV](https://drive.google.com/file/d/1shoCvp_k-M3-8fFh1MyuaQ0bU2GLp3Xp/view?usp=sharing)" )

        uploaded_file_cnn = st.file_uploader("Choose h5 file:", key="file_uploader_2")
        if data_type == "ECG" and new_data is not None:
            uploaded_file = new_data
            st.write(new_data.shape)
            if st.button('Predict'):
                if uploaded_file_cnn is not None:
                    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
                        f.write(uploaded_file_cnn.read())
                        f.seek(0)
                        model_cnn = load_model(f.name)
                        # model_Cnn = load_model("CNN_model.h5")
                        st.caption("Model loaded successfully!!!")
                        predictions = model_cnn.predict(new_data)
                        predictions = np.round(predictions)
                        output = model_cnn.predict(new_data)
                        probabilities = np.round(np.concatenate([1 - output, output], axis=-1), 1)
                        y_classes = probabilities.argmax(axis=-1)  # add this line
                        st.write("Prediction probablities of each target classes: ", probabilities)
                        st.write("Predicted classes: ", y_classes)  # add this line
                        converted_predictions = np.where(predictions == 0, 'N', 'A')
                        # st.write("Predicted Apnea(A)  &  NonApnea(N) ", converted_predictions)
                        st.bar_chart(probabilities)
                        st.warning(
                            "**Above graph shows probability of apnea detected during every minute.In this 0 indicates NonApnea and 1 indicates Apnea.**")
                        if converted_predictions is not None:
                            st.subheader("AHI ( apnea-hypopnea index. )")
                            # code for processing AHI data based on converted_predictions

                            count = 0

                            N_count = 0

                            A_count = 0

                            N_list = []

                            A_list = []

                            for char in converted_predictions:

                                if count == 60:
                                    N_list.append(N_count)

                                    A_list.append(A_count)

                                    N_count = 0

                                    A_count = 0

                                    count = 0

                                count += 1

                                if char == 'A':
                                    A_count += 1

                                if char == 'N':
                                    N_count += 1

                            st.write("Count of Apnea in each hour:", A_list)

                            avrg = np.average(A_list)
                            avrg_rounded = round(avrg, 2)
                            st.write("AVERAGE APNEA SCORE:", avrg_rounded)

                            result = prediction_table(avrg)

                            # st.write("This patient has",result,".")
                            st.markdown(f"This patient has <span style='color:red'>{result}</span>",
                                        unsafe_allow_html=True)

                            st.subheader("HEAT MAP:")
                            data = A_list
                            data = np.array(data).reshape(1, len(data))

                            red_cmap = cm.get_cmap('Reds')
                            colors = red_cmap(np.linspace(0, 1, 4))
                            custom_cmap = matplotlib.colors.ListedColormap(colors)

                            ticks_n = range(1, len(A_list))
                            bounds = [0, 5, 15, 30, 100]
                            norm = matplotlib.colors.BoundaryNorm(bounds, custom_cmap.N)

                            fig = plt.figure(figsize=(2, 1))
                            plt.pcolor(np.transpose(data), cmap=custom_cmap, norm=norm)
                            cb = plt.colorbar(ticks=bounds, boundaries=bounds, values=[0, 5, 15, 30], extend='max',
                                              label='Apnea Scale',
                                              orientation='horizontal', fraction=0.05)
                            cb.ax.tick_params(labelsize=5)
                            cb.ax.set_xlabel('Apnea Scale', fontsize=4)

                            plt.yticks(np.arange(0, data.shape[1]), [f" {i}" for i in (range(0, len(A_list)))])
                            plt.ylabel('Hours', fontsize=3)
                            plt.xticks([], [])

                            plt.gca().tick_params(axis='both', which='major', labelsize=3)

                            st.pyplot(fig)
                            st.warning("**Above graph shows heatmap of no of Apneas in each hour.**")

        elif data_type == "HRV" and data is not None:
            uploaded_file = data
            st.write(data.shape)
            if st.button('Predict'):
                if uploaded_file_cnn is not None:
                    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
                        f.write(uploaded_file_cnn.read())
                        f.seek(0)
                        model_cnn = load_model(f.name)
                        # model_Cnn = load_model("CNN_model.h5")
                        st.caption("Model loaded successfully!!!")
                        predictions = model_cnn.predict(data)
                        predictions = np.round(predictions)
                        output = model_cnn.predict(data)
                        probabilities = np.round(np.concatenate([1 - output, output], axis=-1), 1)
                        y_classes = probabilities.argmax(axis=-1)  # add this line
                        st.write("Prediction probablities of each target classes: ", probabilities)
                        st.write("Predicted classes: ", y_classes)  # add this line
                        converted_predictions = np.where(predictions == 0, 'N', 'A')
                        # st.write("Predicted Apnea(A)  &  NonApnea(N) ", converted_predictions)
                        st.bar_chart(probabilities)
                        st.warning(
                            "**Above graph shows probability of apnea detected during every minute.In this 0 indicates NonApnea and 1 indicates Apnea.**")
                        if converted_predictions is not None:
                            st.subheader("AHI ( apnea-hypopnea index. )")
                            # code for processing AHI data based on converted_predictions

                            count = 0

                            N_count = 0

                            A_count = 0

                            N_list = []

                            A_list = []

                            for char in converted_predictions:

                                if count == 60:
                                    N_list.append(N_count)

                                    A_list.append(A_count)

                                    N_count = 0

                                    A_count = 0

                                    count = 0

                                count += 1

                                if char == 'A':
                                    A_count += 1

                                if char == 'N':
                                    N_count += 1

                            st.write("Count of Apnea in each hour:", A_list)

                            avrg = np.average(A_list)
                            avrg_rounded = round(avrg, 2)
                            st.write("AVERAGE APNEA SCORE:", avrg_rounded)

                            result = prediction_table(avrg)

                            # st.write("This patient has",result,".")
                            st.markdown(f"This patient has <span style='color:red'>{result}</span>",
                                        unsafe_allow_html=True)

                            st.subheader("HEAT MAP:")
                            dataa = A_list
                            dataa = np.array(dataa).reshape(1, len(dataa))

                            red_cmap = cm.get_cmap('Reds')
                            colors = red_cmap(np.linspace(0, 1, 4))
                            custom_cmap = matplotlib.colors.ListedColormap(colors)

                            ticks_n = range(1, len(A_list))
                            bounds = [0, 5, 15, 30, 100]
                            norm = matplotlib.colors.BoundaryNorm(bounds, custom_cmap.N)

                            fig = plt.figure(figsize=(2, 1))
                            plt.pcolor(np.transpose(dataa), cmap=custom_cmap, norm=norm)
                            cb = plt.colorbar(ticks=bounds, boundaries=bounds, values=[0, 5, 15, 30], extend='max',
                                              label='Apnea Scale',
                                              orientation='horizontal', fraction=0.05)
                            cb.ax.tick_params(labelsize=5)
                            cb.ax.set_xlabel('Apnea Scale', fontsize=4)

                            plt.yticks(np.arange(0, dataa.shape[1]), [f" {i}" for i in (range(0, len(A_list)))])
                            plt.ylabel('Hours', fontsize=3)
                            plt.xticks([], [])

                            plt.gca().tick_params(axis='both', which='major', labelsize=3)

                            st.pyplot(fig)
                            st.warning("**Above graph shows heatmap of no of Apneas in each hour.**")



    #else:

            #st.write(
                #"Please select the CNN option and click 'Predict' to generate annotations before calculating AHI.")


if __name__ == "__main__":
    sleep()



