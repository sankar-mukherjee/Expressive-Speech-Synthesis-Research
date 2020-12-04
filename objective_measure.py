import glob
import os
import pickle
import re
from pesq import pesq
from pystoi import stoi

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
from multiprocessing import Pool
import functools

SPEECH_RECOGNIZE = sr.Recognizer()


def stoi_score(original, synthesis):
    distance, path = fastdtw(original, synthesis, dist=euclidean)
    distance /= (len(original) + len(synthesis))
    path_x = list(map(lambda l: l[0], path))
    path_y = list(map(lambda l: l[1], path))
    score = stoi(original[path_x], synthesis[path_y], 16000, extended=False)
    return score


def pesq_score(original, synthesis):
    distance, path = fastdtw(original, synthesis, dist=euclidean)
    distance /= (len(original) + len(synthesis))
    path_x = list(map(lambda l: l[0], path))
    path_y = list(map(lambda l: l[1], path))
    score = pesq(16000, original[path_x], synthesis[path_y], 'wb')
    return score


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
        matches = re.search(r"\d+", x)
        num = x[matches.start(0):matches.end(0)]
        num = num2words(num)
        x = x[:matches.start(0)] + ' ' + num + ' ' + x[matches.end(0):]
    # symbol

    for sym in replace_symbols:
        x = x.replace(sym, replace_symbols[sym])
    # remove multiple white spaces inside string
    x = re.sub(' +', ' ', x)
    return x


def compute_scores(ori_wav, syn, sr_syn, original_f0, mgc_original, syn_file_name, ori_file_name, original_text):
    try:
        # PESQ (Perceptual Evaluation of Speech Quality)
        tmp_pesq = pesq_score(ori_wav, syn)
        # Short Term Objective Intelligibility
        temp_stoi = stoi_score(ori_wav, syn)
        # Root Mean Squared Error F0
        f0_synthesis, _ = pyworld.harvest(syn.astype(np.float64), sr_syn, frame_period=5, f0_floor=70.0, f0_ceil=800.0)
        tmp_rmse = RMSE_f0(original_f0, f0_synthesis)
        # Mel-generalized cepstrum
        synthesis_mgc = readmgc(syn)
        # Mel-cepstral distortion (MCD)
        tmp_mcd = MCD(mgc_original, synthesis_mgc)
        # Frame Disturbance (FD)
        tmp_fd = FD(mgc_original, synthesis_mgc)
        # Word error rate
        predicted_syn_txt = recognize_speech(syn_file_name)
        predicted_ori_txt = recognize_speech(ori_file_name)

        wer_predicted_ori_predicted_syn = jiwer.wer(text_normalization(original_text),
                                                    text_normalization(predicted_syn_txt))
        wer_ori_predicted_ori = jiwer.wer(text_normalization(original_text), text_normalization(predicted_ori_txt))
        print(syn_file_name)
    except:
        tmp_rmse, tmp_mcd, tmp_fd, tmp_pesq, temp_stoi, \
        wer_predicted_ori_predicted_syn, wer_ori_predicted_ori = 0, 0, 0, 0, 0, 0, 0
        print('bad file: ' + syn_file_name)

    df = pd.DataFrame({'RMSE_F0': [round(tmp_rmse, 3)],
                       'MCD': round(tmp_mcd, 3),
                       'FD': round(tmp_fd, 3),
                       'PESQ': round(tmp_pesq, 3),
                       'Stoi': round(temp_stoi, 3),
                       'WER_pred_ori_pred_syn': round(wer_predicted_ori_predicted_syn, 3),
                       'WER_ori_pred_ori': round(wer_ori_predicted_ori, 3)})
    print(df)
    return df


def get_original_scores(audio_path, txt_path):
    ref_file_names = glob.glob(audio_path + "*.wav")
    if not os.path.exists(audio_path + 'metrics.npy'):
        file_id = open(txt_path + 'test_sentences', "r")
        test_sentence = list(file_id)
        file_id.close()
        text = []
        for i, sentence in enumerate(test_sentence):
            sen = sentence.split('|')
            text.append({'id': sen[0], 'txt': sen[1].rstrip()})

        metrics_data = []
        for file_id in ref_file_names:
            temp = file_id.replace(audio_path, '')
            ori, sr = librosa.load(audio_path + temp, sr=None)
            # Mel-generalized cepstrum
            original_mgc = readmgc(ori)
            f0_original, _ = pyworld.harvest(ori.astype(np.float64), sr, frame_period=5, f0_floor=70.0, f0_ceil=800.0)

            tmp_text = list(filter(lambda x: x['id'] == temp.replace('.wav', ''), text))

            metrics_data.append({'file_id': temp, 'wav': ori, 'mgc': original_mgc, 'f0': f0_original,
                                 'txt': tmp_text[0]['txt']})
            print(file_id)
        np.save(audio_path + 'metrics.npy', metrics_data)
        ori_metrics = np.load(audio_path + 'metrics.npy', allow_pickle=True)
    else:
        ori_metrics = np.load(audio_path + 'metrics.npy', allow_pickle=True)
    return ori_metrics


def get_same_combination_scores(audio_path, out_path, original_metric):
    syn_file_names = glob.glob(out_path + '/for_objective_measurement/' + "*.wav")

    if not os.path.exists(out_path + '_score.csv'):
        metrics_data = []
        for syn_file_name in syn_file_names:
            temp = syn_file_name.replace(out_path + '/for_objective_measurement/', '')
            syn, sr_syn = librosa.load(syn_file_name, sr=None)

            # get text id
            text_id = temp.split('-')[1]

            ori_metrics = list(filter(lambda x: x['file_id'] == text_id, original_metric))[0]
            ori_wav = ori_metrics['wav']
            mgc_original = ori_metrics['mgc']
            original_f0 = ori_metrics['f0']
            original_text = ori_metrics['txt']

            ori_file_name = audio_path + text_id + '.wav'
            temp = compute_scores(ori_wav, syn, sr_syn, original_f0, mgc_original,
                                  syn_file_name, ori_file_name, original_text)
            metrics_data.append(temp)
        metrics_data = pd.concat(metrics_data, axis=0)
        metrics_data.to_csv(out_path + '_score.csv', index=False)


def get_rand_combination_scores(out_path, original_metric):
    if not os.path.exists(out_path + '_score_rand.csv'):
        syn_file_names = glob.glob(out_path + '/for_objective_measurement_rand/' + "*.wav")
        metrics_data = []
        for syn_file_name in syn_file_names:
            temp = syn_file_name.replace(out_path + '/for_objective_measurement_rand/', '')
            syn, sr_syn = librosa.load(syn_file_name, sr=None)

            # get style
            style_id = temp.split(':')[1].split('-')[1]
            ori_metrics = list(filter(lambda x: x['file_id'] == style_id + '.wav', original_metric))[0]
            ori_wav = ori_metrics['wav']
            mgc_original = ori_metrics['mgc']
            original_f0 = ori_metrics['f0']

            # get text id
            text_id = temp.split(':')[0].split('-')[1]
            ori_metrics = list(filter(lambda x: x['file_id'] == text_id + '.wav', original_metric))[0]
            original_text = ori_metrics['txt']

            ori_file_name = wav_path + text_id + '.wav'
            temp = compute_scores(ori_wav, syn, sr_syn, original_f0, mgc_original,
                                  syn_file_name, ori_file_name, original_text)
            metrics_data.append(temp)
        metrics_data = pd.concat(metrics_data, axis=0)
        metrics_data.to_csv(out_path + '_score_rand.csv', index=False)


def get_text_rand_combination_scores(out_path, original_metric):
    if not os.path.exists(out_path + '_score_text_rand.csv'):
        syn_file_names = glob.glob(out_path + '/for_objective_measurement_text_rand/' + "*.wav")
        metrics_data = []
        for syn_file_name in syn_file_names:
            temp = syn_file_name.replace(out_path + '/for_objective_measurement_text_rand/', '')
            syn, sr_syn = librosa.load(syn_file_name, sr=None)

            # get style
            style_id = temp.split(':')[1].split('-')[1]

            ori_metrics = list(filter(lambda x: x['file_id'] == style_id + '.wav', original_metric))[0]
            ori_wav = ori_metrics['wav']
            mgc_original = ori_metrics['mgc']
            original_f0 = ori_metrics['f0']
            original_text = ori_metrics['txt']

            ori_file_name = wav_path + style_id + '.wav'
            temp = compute_scores(ori_wav, syn, sr_syn, original_f0, mgc_original,
                                  syn_file_name, ori_file_name, original_text)
            metrics_data.append(temp)
        metrics_data = pd.concat(metrics_data, axis=0)
        metrics_data.to_csv(out_path + '_score_text_rand.csv', index=False)


def get_style_rand_combination_scores(out_path, original_metric):
    if not os.path.exists(out_path + '_score_style_rand.csv'):
        syn_file_names = glob.glob(out_path + '/for_objective_measurement_style_rand/' + "*.wav")
        metrics_data = []
        for syn_file_name in syn_file_names:
            temp = syn_file_name.replace(out_path + '/for_objective_measurement_style_rand/', '')
            syn, sr_syn = librosa.load(syn_file_name, sr=None)

            # get text id
            text_id = temp.split(':')[0].split('-')[1]

            ori_metrics = list(filter(lambda x: x['file_id'] == text_id + '.wav', original_metric))[0]
            ori_wav = ori_metrics['wav']
            mgc_original = ori_metrics['mgc']
            original_f0 = ori_metrics['f0']
            original_text = ori_metrics['txt']

            ori_file_name = wav_path + text_id + '.wav'
            temp = compute_scores(ori_wav, syn, sr_syn, original_f0, mgc_original,
                                  syn_file_name, ori_file_name, original_text)
            metrics_data.append(temp)
        metrics_data = pd.concat(metrics_data, axis=0)
        metrics_data.to_csv(out_path + '_score_style_rand.csv', index=False)


def func_map(func):
    return func()


if __name__ == "__main__":
    # do objective measurement sr = 16000
    # no of parallel process
    no_process = 30
    output_path = 'output/'
    model_names = {'wavernn_gst', 'wavernn_gst_mine_concat', 'wavernn_gst_H-mine', 'wavernn_gst_R-mine'}
    model_names = {'S23_ckpt-40_wavernn_test_mine', 'S24_ckpt-40_wavernn_test_mine', 'S25_ckpt-40_wavernn_test_mine',
                   'S26_ckpt-40_wavernn_test_mine', 'S27_ckpt-40_wavernn_test_mine', 'S28_ckpt-40_wavernn_test_mine',
                   'S29_ckpt-40_wavernn_test_mine', 'S30_ckpt-40_wavernn_test_mine', 'S31_ckpt-40_wavernn_test_mine',
                   'S32_ckpt-40_wavernn_test_mine', 'S33_ckpt-40_wavernn_test_mine', 'S34_ckpt-40_wavernn_test_mine',
                   'S35_ckpt-40_wavernn_test_mine', 'S36_ckpt-40_wavernn_test_mine', 'S37_ckpt-40_wavernn_test_mine',
                   'S38_ckpt-40_wavernn_test_mine', 'S39_ckpt-40_wavernn_test_mine', 'S40_ckpt-40_wavernn_test_mine'}

    wav_path = '../database/ref_audio/for_speaker_tts/style/'
    text_path = '../database/ref_audio/'
    original_metrics = get_original_scores(wav_path, text_path)

    funcs = []
    for m in model_names:
        f_same = functools.partial(get_same_combination_scores, wav_path, output_path + m, original_metrics)
        f_rand = functools.partial(get_rand_combination_scores, output_path + m, original_metrics)
        f_text_rand = functools.partial(get_text_rand_combination_scores, output_path + m, original_metrics)
        f_style_rand = functools.partial(get_style_rand_combination_scores, output_path + m, original_metrics)
        funcs.append([f_same, f_rand, f_text_rand, f_style_rand])
    funcs = [item for sublist in funcs for item in sublist]

    # Create a multiprocessing Pool
    with Pool(no_process) as pool:
        res = pool.map(func_map, funcs)

    ####################################################################################################################
    # all scores
    score_file_names = glob.glob(output_path + "*_score.csv")
    metrics = []
    for f in score_file_names:
        tmp = pd.read_csv(f)
        tmp = tmp.mean(axis=0).round(3)
        model_name = pd.Series({'model': f.replace(output_path, '').replace('_score.csv', '')})
        metrics.append(pd.DataFrame(tmp.append(model_name)))
    metrics = pd.concat(metrics, axis=1).transpose()
    metrics.to_csv(output_path + 'all_score.log', index=False, header=True, sep='\t')

    # all scores rand
    score_file_names = glob.glob(output_path + "*_score_rand.csv")
    metrics = []
    for f in score_file_names:
        tmp = pd.read_csv(f)
        tmp = tmp.mean(axis=0).round(3)
        model_name = pd.Series({'model': f.replace(output_path, '').replace('_score_rand.csv', '')})
        metrics.append(pd.DataFrame(tmp.append(model_name)))
    metrics = pd.concat(metrics, axis=1).transpose()
    metrics.to_csv(output_path + 'all_score_rand.log', index=False, header=True, sep='\t')
    metrics[['WER_pred_ori_pred_syn', 'model']].to_csv(output_path + 'eval_score_rand.log', index=False, header=True,
                                                       sep='\t')

    # all scores text rand
    score_file_names = glob.glob(output_path + "*_score_text_rand.csv")
    metrics = []
    for f in score_file_names:
        tmp = pd.read_csv(f)
        tmp = tmp.mean(axis=0).round(3)
        model_name = pd.Series({'model': f.replace(output_path, '').replace('_score_text_rand.csv', '')})
        metrics.append(pd.DataFrame(tmp.append(model_name)))
    metrics = pd.concat(metrics, axis=1).transpose()
    metrics.to_csv(output_path + 'all_score_text_rand.log', index=False, header=True, sep='\t')
    metrics[['RMSE_F0', 'model']].to_csv(output_path + 'eval_score_text_rand.log', index=False, header=True, sep='\t')

    # all scores style rand
    score_file_names = glob.glob(output_path + "*_score_style_rand.csv")
    metrics = []
    for f in score_file_names:
        tmp = pd.read_csv(f)
        tmp = tmp.mean(axis=0).round(3)
        model_name = pd.Series({'model': f.replace(output_path, '').replace('_score_style_rand.csv', '')})
        metrics.append(pd.DataFrame(tmp.append(model_name)))
    metrics = pd.concat(metrics, axis=1).transpose()
    metrics.to_csv(output_path + 'all_score_style_rand.log', index=False, header=True, sep='\t')
    metrics[['MCD', 'FD', 'PESQ', 'Stoi', 'model']].to_csv(output_path + 'eval_score_style_rand.log', index=False,
                                                           header=True,
                                                           sep='\t')
