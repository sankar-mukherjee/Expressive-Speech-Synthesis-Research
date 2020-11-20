import glob
import os
import pickle
import re

import librosa
import numpy as np
import pysptk
from num2words import num2words
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pyworld
import speech_recognition as sr
import jiwer
import pandas as pd
import matplotlib.pyplot as plt

SPEECH_RECOGNIZE = sr.Recognizer()


def readmgc(x):
    # Mel-generalized cepstrum
    x = x.astype(np.float64)
    frame_length = 1024
    hop_length = 200
    # Windowing
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames *= pysptk.blackman(frame_length)
    assert frames.shape[1] == frame_length
    # Order of mel-cepstrum
    order = 20  # order = 20 sr = 16000, order = 25 for sr = 22050
    alpha = 0.41
    stage = 5
    gamma = -1.0 / stage

    mgc = pysptk.mgcep(frames, order, alpha, gamma, min_det=1.0e-21)
    mgc = mgc.reshape(-1, order + 1)
    return mgc


def MCD(original, synthesis):
    # Mel-cepstral distortion (MCD)
    distance, path = fastdtw(original, synthesis, dist=euclidean)
    distance /= (len(original) + len(synthesis))
    path_x = list(map(lambda l: l[0], path))
    path_y = list(map(lambda l: l[1], path))

    mcd = np.sum(np.square(original[path_x] - synthesis[path_y]), axis=0)
    mcd = np.sqrt(mcd) / float(original[path_x].shape[0])
    mcd = (10.0 / np.log(10.0) * np.sqrt(2.0)) * mcd
    mcd = np.sum(mcd)
    return mcd


def FD(original, synthesis):
    # Frame Disturbance (FD)
    distance, path = fastdtw(original, synthesis, dist=euclidean)
    distance /= (len(original) + len(synthesis))
    path_x = list(map(lambda l: l[0], path))
    path_y = list(map(lambda l: l[1], path))

    fd = np.sqrt(np.mean(np.square(original[path_x] - synthesis[path_y])))
    return fd


def RMSE_f0(original, synthesis):
    # normalize 0 - 1
    # original = (original - np.min(original))/(np.max(original)-np.min(original))
    # synthesis = (synthesis - np.min(synthesis))/(np.max(synthesis)-np.min(synthesis))
    # Root Mean Squared Error F0
    distance, path = fastdtw(original, synthesis, dist=euclidean)
    distance /= (len(original) + len(synthesis))
    path_x = list(map(lambda l: l[0], path))
    path_y = list(map(lambda l: l[1], path))
    rmse_f0 = np.sqrt(np.mean(np.square(original[path_x] - synthesis[path_y])))
    return rmse_f0


def recognize_speech(audio_file):
    try:
        with sr.AudioFile(audio_file) as source:
            audio = SPEECH_RECOGNIZE.record(source)
            txt = SPEECH_RECOGNIZE.recognize_google(audio, language="en-GB")
        return txt
    except:
        return ''


replace_symbols = {'(': ' ',
                   ')': ' ',
                   '-': ' ',
                   '...': '.',
                   '..': '.',
                   '--': ' ',
                   ';': '',
                   ',': '',
                   '.': ''}


def text_normalization(x):
    # general
    x = x.rstrip().lower()
    # numbers to words
    while bool(re.search(r'\d', x)):
        m = re.search(r"\d+", x)
        num = x[m.start(0):m.end(0)]
        num = num2words(num)
        x = x[:m.start(0)] + ' ' + num + ' ' + x[m.end(0):]
    # symbol

    for i in replace_symbols:
        x = x.replace(i, replace_symbols[i])
    # remove multiple white spaces inside string
    x = re.sub(' +', ' ', x)
    return x


if __name__ == "__main__":
    # do objective measurement sr = 16000
    out_path = 'output/'
    model_names = {'wavernn_gst', 'wavernn_gst_mine_concat', 'wavernn_gst_H-mine', 'wavernn_gst_R-mine'}
    model_names = {'S30_ckpt-40_wavernn_test_mine'}

    wav_path = '../database/ref_audio/for_speaker_tts/style/'
    text_path = '../database/ref_audio/'
    ref_file_names = glob.glob(wav_path + "*.wav")

    if not os.path.exists(wav_path + 'metrics.npy'):
        f = open(text_path + 'test_sentences', "r")
        test_sentence = list(f)
        f.close()
        text = []
        for i, sentence in enumerate(test_sentence):
            sen = sentence.split('|')
            text.append({'id': sen[0], 'txt': sen[1].rstrip()})

        metrics = []
        for f in ref_file_names:
            tmp = f.replace(wav_path, '')
            ori, sr = librosa.load(wav_path + tmp, sr=None)
            # Mel-generalized cepstrum
            original_mgc = readmgc(ori)
            f0_original, _ = pyworld.harvest(ori.astype(np.float64), sr, frame_period=5, f0_floor=70.0, f0_ceil=800.0)

            tmp_text = list(filter(lambda x: x['id'] == tmp.replace('.wav', ''), text))

            metrics.append({'file_id': tmp, 'wav': ori, 'mgc': original_mgc, 'f0': f0_original,
                            'txt': tmp_text[0]['txt']})
            print(f)
        np.save(wav_path + 'metrics.npy', metrics)
        original_metrics = np.load(wav_path + 'metrics.npy', allow_pickle=True)
    else:
        original_metrics = np.load(wav_path + 'metrics.npy', allow_pickle=True)

    for m in model_names:
        if not os.path.exists(out_path + m + '_score.csv'):
            metrics = []
            for f in ref_file_names:
                tmp = f.replace(wav_path, '')
                file_name = out_path + m + '/for_objective_measurement/synthesized-' + tmp
                syn, sr_syn = librosa.load(file_name, sr=None)
                ori_metrics = list(filter(lambda x: x['file_id'] == tmp, original_metrics))[0]
                ori = ori_metrics['wav']
                original_mgc = ori_metrics['mgc']
                f0_original = ori_metrics['f0']
                original_text = ori_metrics['txt']

                tmp_rmse = 0
                tmp_mcd = 0
                tmp_fd = 0
                wer_predicted_ori_predicted_syn = 0
                wer_ori_predicted_ori = 0
                try:
                    # Root Mean Squared Error F0
                    f0_synthesis, _ = pyworld.harvest(syn.astype(np.float64), sr_syn, frame_period=5,
                                                      f0_floor=70.0,
                                                      f0_ceil=800.0)
                    tmp_rmse = RMSE_f0(f0_original, f0_synthesis)
                    # Mel-generalized cepstrum
                    synthesis_mgc = readmgc(syn)
                    # Mel-cepstral distortion (MCD)
                    tmp_mcd = MCD(original_mgc, synthesis_mgc)
                    # Frame Disturbance (FD)
                    tmp_fd = FD(original_mgc, synthesis_mgc)
                    # Word error rate
                    predicted_syn_txt = recognize_speech(file_name)
                    predicted_ori_txt = recognize_speech(f)

                    wer_predicted_ori_predicted_syn = jiwer.wer(text_normalization(original_text),
                                                                text_normalization(predicted_syn_txt))
                    wer_ori_predicted_ori = jiwer.wer(text_normalization(original_text),
                                                      text_normalization(predicted_ori_txt))

                    tmp = pd.DataFrame({'RMSE_F0': [round(tmp_rmse, 3)],
                                                 'MCD': round(tmp_mcd, 3),
                                                 'FD': round(tmp_fd, 3),
                                                 'WER_pred_ori_pred_syn': round(wer_predicted_ori_predicted_syn, 3),
                                                 'WER_ori_pred_ori': round(wer_ori_predicted_ori, 3)})
                    metrics.append(tmp)
                    print(file_name)
                    print(tmp)
                except:
                    print('bad file: ' + file_name)
            metrics = pd.concat(metrics, axis=0)
            metrics.to_csv(out_path + m + '_score.csv', index=False)

        if not os.path.exists(out_path + m + '_score_rand.csv'):
            syn_file_names = glob.glob(out_path + m + '/for_objective_measurement_rand/' + "*.wav")
            metrics = []
            for f in syn_file_names:
                tmp = f.replace(out_path + m + '/for_objective_measurement_rand/', '')
                syn, sr_syn = librosa.load(f, sr=None)

                # get style and text
                style_id = tmp.split(':')[1].split('-')[1]
                text_id = tmp.split(':')[0].split('-')[1]

                ori_metrics = list(filter(lambda x: x['file_id'] == style_id+'.wav', original_metrics))[0]
                ori = ori_metrics['wav']
                original_mgc = ori_metrics['mgc']
                f0_original = ori_metrics['f0']

                ori_metrics = list(filter(lambda x: x['file_id'] == text_id+'.wav', original_metrics))[0]
                original_text = ori_metrics['txt']

                tmp_rmse = 0
                tmp_mcd = 0
                tmp_fd = 0
                wer_predicted_ori_predicted_syn = 0
                wer_ori_predicted_ori = 0
                try:
                    # Root Mean Squared Error F0
                    f0_synthesis, _ = pyworld.harvest(syn.astype(np.float64), sr_syn, frame_period=5,
                                                      f0_floor=70.0,
                                                      f0_ceil=800.0)
                    tmp_rmse = RMSE_f0(f0_original, f0_synthesis)
                    # Mel-generalized cepstrum
                    synthesis_mgc = readmgc(syn)
                    # Mel-cepstral distortion (MCD)
                    tmp_mcd = MCD(original_mgc, synthesis_mgc)
                    # Frame Disturbance (FD)
                    tmp_fd = FD(original_mgc, synthesis_mgc)
                    # Word error rate
                    predicted_syn_txt = recognize_speech(f)
                    predicted_ori_txt = recognize_speech(wav_path+text_id+'.wav')

                    wer_predicted_ori_predicted_syn = jiwer.wer(text_normalization(original_text),
                                                                text_normalization(predicted_syn_txt))
                    wer_ori_predicted_ori = jiwer.wer(text_normalization(original_text),
                                                      text_normalization(predicted_ori_txt))

                    tmp = pd.DataFrame({'RMSE_F0': [round(tmp_rmse, 3)],
                                                 'MCD': round(tmp_mcd, 3),
                                                 'FD': round(tmp_fd, 3),
                                                 'WER_pred_ori_pred_syn': round(wer_predicted_ori_predicted_syn, 3),
                                                 'WER_ori_pred_ori': round(wer_ori_predicted_ori, 3)})
                    metrics.append(tmp)
                    print(f)
                    print(tmp)
                except:
                    print('bad file: ' + f)
            metrics = pd.concat(metrics, axis=0)
            metrics.to_csv(out_path + m + '_score_rand.csv', index=False)

    # all scores
    score_file_names = glob.glob(out_path + "*_score.csv")
    metrics = []
    for f in score_file_names:
        tmp = pd.read_csv(f)
        tmp = tmp.mean(axis=0).round(3)
        model_name = pd.Series({'model': f.replace(out_path, '').replace('_score.csv', '')})
        metrics.append(pd.DataFrame(tmp.append(model_name)))
    metrics = pd.concat(metrics, axis=1).transpose()
    metrics.to_csv(out_path + 'all_score.log', index=False, header=True, sep='\t')

    # all scores rand
    score_file_names = glob.glob(out_path + "*_score_rand.csv")
    metrics = []
    for f in score_file_names:
        tmp = pd.read_csv(f)
        tmp = tmp.mean(axis=0).round(3)
        model_name = pd.Series({'model': f.replace(out_path, '').replace('_score_rand.csv', '')})
        metrics.append(pd.DataFrame(tmp.append(model_name)))
    metrics = pd.concat(metrics, axis=1).transpose()
    metrics.to_csv(out_path + 'all_score_rand.log', index=False, header=True, sep='\t')
