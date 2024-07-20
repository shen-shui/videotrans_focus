import os
import re
import subprocess

import pandas as pd
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchaudio

from videotrans.tts import text_to_speech
from videotrans.util import tools
from pathlib import Path
import numpy as np
import librosa

from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment, Annotation

from scipy.spatial.distance import cosine
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
# 确保这些属性存在于模型中
if hasattr(model.encoder.pos_conv_embed.conv, 'parametrizations') and \
   hasattr(model.encoder.pos_conv_embed.conv.parametrizations, 'weight'):

    # 初始化weight.original0和weight.original1
    if hasattr(model.encoder.pos_conv_embed.conv.parametrizations.weight, 'original0'):
        init.xavier_uniform_(model.encoder.pos_conv_embed.conv.parametrizations.weight.original0)

    if hasattr(model.encoder.pos_conv_embed.conv.parametrizations.weight, 'original1'):
        init.xavier_uniform_(model.encoder.pos_conv_embed.conv.parametrizations.weight.original1)

# if hasattr(model, 'masked_spec_embed'):
#     init.xavier_uniform_(model.masked_spec_embed)


homepath=Path.home()/'Videos/pyvideotrans/tts'
homedir = homepath.as_posix()
wav_dir = os.path.join(homedir, 'wav')
embedding_path=os.path.join(wav_dir,"role_embeddings.npy")

if not os.path.exists(homedir):
    os.makedirs(homedir, exist_ok=True)
if not os.path.exists(wav_dir):
    os.makedirs(wav_dir, exist_ok=True)

# 取多个segment中的特征值
def extract_embedding_from_segments(embedding_model, audio_file, segments):
    embeddings = []
    y, sr = librosa.load(audio_file, sr=None)
    for segment in segments:
        # 修正片段的结束时间，确保其不超过音频文件的实际长度
        start = segment.start
        end = min(segment.end, len(y) / sr)
        if start >= end:
            continue
        waveform = y[int(start * sr):int(end * sr)]
        
         # 将 numpy 数组转换为 PyTorch 张量，并添加一个额外的维度，以适应模型输入
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        
        # 调整样本率，某些模型可能需要特定的采样率
        sample_rate = sr
        if hasattr(embedding_model, 'sample_rate') and embedding_model.sample_rate != sample_rate:
            waveform_tensor = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=embedding_model.sample_rate)(waveform_tensor)
            sample_rate = embedding_model.sample_rate
        
        # 现在可以调用模型了
        with torch.no_grad():
            embedding = embedding_model(waveform_tensor)
        
        # 假设模型输出是一个 batch 的嵌入，我们只取第一个元素
        embeddings.append(embedding.squeeze(0).numpy())
        
    return np.mean(embeddings, axis=0)  # 对所有有声片段的嵌入取平均

# 提取音频片段的embedding
def get_embedding(embedding_model, audio_path, segment):
    # 先直接用whole吧，简单方便
    inference = Inference(embedding_model, window="whole") 
    embedding = inference.crop(audio_path, segment) 

    # embedding = get_embedding_values(sliding_window_feature)
    return embedding

def get_embedding_for_vec(audio_path, segment):
    audio , _ =librosa.load(audio_path, sr=16000, offset= segment.start, duration=segment.duration)    
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        output = model(input_values).last_hidden_state.mean(dim=1)

        return output[0].numpy()

# for sliding window
# def get_embedding_values(sliding_window_feature):
#     if isinstance(sliding_window_feature, tuple):
#         # 处理元组的情况
#         return tuple(e.data for e in sliding_window_feature)
#     else:
#         # 处理单个 SlidingWindowFeature 的情况
#         return sliding_window_feature.data


# 初始化语音活动检测和嵌入模型
vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token='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')
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token='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')

# 提取音频中有声片段segment
def extract_voiced_segments(audio_file):
    # 使用 VAD 提取有声片段
    vad_result = vad_pipeline(audio_file)

    y, sr = librosa.load(audio_file, sr=None)
    duration = len(y) / sr

    # 修正片段的结束时间，确保其不超过音频文件的实际长度
    voiced_segments = []
    for segment in vad_result.get_timeline().support():
        start = segment.start
        end = min(segment.end, duration)
        if start < end:
            voiced_segments.append(Segment(start, end))
    return voiced_segments
    # voiced_segments = [segment for segment, _, label in vad_result.itertracks(yield_label=True) if label == 'SPEECH']
    # return voiced_segments

# 提取音频的整段segment
def extract_audio_segment(audio_file):
    y, sr= librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    # 取整个语音片段
    segment = Segment(0, duration)

    return segment

def save_embeddings(role_audio_files, embedding_model, save_path=embedding_path):
    embeddings = {}
    for role, audio_file in role_audio_files.items():
        # embedding = get_embedding_from_audio(audio_file)
        embedding = get_vec_embedding(audio_file)
        # 取音频中有声音的部分，计算特征值，效果待验证
        # embedding = get_voiced_embedding_from_audio(audio_file)
        embeddings[role] = embedding
    np.save(save_path, embeddings)

# 提取并保存配音角色的特征值
def extract_embeddings(save_embeddings):
    save_embeddings(get_role_wav_files(), embedding_model)

def get_role_mp3_files(dir = homedir):
    mp3_files = [f for f in os.listdir(dir) if f.endswith('.mp3')]
    return mp3_files

#  
#   返回格式如下：
#   {
#      "en-IN-NeerjaNeural": "path/to/en-IN-NeerjaNeural.wav",
#      "en-NZ-MitchellNeural": "path/to/en-NZ-MitchellNeural.wav"
#   }
#  
def get_role_wav_files():
    # 获取所有以.wav结尾的配音文件
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    # 创建一个字典，键是角色名称，值是文件路径
    role_wav_dict = {}
    for wav_file in wav_files:
        # 角色名称是从文件名字符中提取
        role_name = wav_file.replace('.wav', '')
        full_path = os.path.join(wav_dir, wav_file)
        
        role_wav_dict[role_name] = full_path
    
    return role_wav_dict

def convert_to_wav(file_path, out_path):
    command = [
        'ffmpeg',
        '-i', file_path,
        '-ac', '1',  # 设置声道数为1（单声道）
        '-ar', '16000',  # 设置采样率为44100Hz
        '-vn',  # 忽略视频流
        '-c:a', 'pcm_s16le',  # 设置音频编码为16位线性PCM
        out_path
    ]
    subprocess.run(command, check=True)

from scipy.spatial.distance import cdist
def calculate_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2) # 相似度值，越接近1越相似

def find_most_similar_role(new_audio_embedding, saved_embeddings):
    similarities = {}
    for role, embedding in saved_embeddings.items():
        similarity = calculate_similarity(new_audio_embedding, embedding)
        similarities[role] = similarity
        # print(f"Similarity between {role} and new audio: {similarity}")
    most_similar_role = max(similarities, key=similarities.get)
    return most_similar_role, similarities

# 匹配一个最合适的角色
def get_role(audio_embedding):
    # 加载预先保存的配音角色特征值
    saved_embeddings = np.load(embedding_path, allow_pickle=True).item()
    role, similarity = find_most_similar_role(audio_embedding, saved_embeddings)

    print(f"The most similar role is {role} with similarity {similarity[role]}")
    return role

def get_embedding_from_audio(audio_file):
    inference = Inference(embedding_model, window="whole") 
    embedding = inference(audio_file)
    # segment = extract_audio_segment(audio_file)
    # embedding = get_embedding(embedding_model, audio_file, segment)
    return embedding

def get_voiced_embedding_from_audio(audio_file):
    segments = extract_voiced_segments(audio_file)
    embedding = extract_embedding_from_segments(embedding_model, audio_file, segments)
    return embedding

# 从所有wav配音文件中找到跟embedding最匹配的
def find_from_roles(embedding):
    similarities = {}
    audio_files = get_role_wav_files()
    for role, audio_file in audio_files.items():
        audio_embedding = concact_embedding(get_embedding_from_audio(audio_file), get_vec_embedding(audio_file))
        # audio_embedding = get_vec_embedding(audio_file)
        # audio_embedding = get_embedding_from_audio(audio_file)
        # 取音频中有声音的部分，计算特征值，效果待验证
        # embedding = get_voiced_embedding_from_audio(audio_file)

        similarity = calculate_similarity(embedding, audio_embedding)
        similarities[role] = similarity
        # print(f"Similarity between {role} and new audio: {similarity}")
    most_similar_role = max(similarities, key=similarities.get)
    return most_similar_role, similarities

# 算出所有wav配音文件的embedding
def get_role_wav_embeddings():
    audio_files = get_role_wav_files()
    embeddings = {}
    for role, audio_file in audio_files.items():
        embedding = concact_embedding(get_embedding_from_audio(audio_file), get_vec_embedding(audio_file))
        embeddings[role] = embedding
    return embeddings

# 俩音频的相似度值
def get_similar(audio_file1, audio_file2):

    similarity = calculate_similarity(get_embedding_from_audio(audio_file1), get_embedding_from_audio(audio_file2))    
    
    # similarity = calculate_similarity(get_voiced_embedding_from_audio(audio_file1), get_voiced_embedding_from_audio(audio_file2))
    return similarity

# 生成所有edge的角色配音文件，方便调试
def make_role_audio():
    speak_text = 'I hope your every day is beautiful and enjoyable!'
    role_list = tools.get_edge_rolelist()
    for r in list(role_list['en']):
        if r == 'No':
            continue
        wavname = f"{homedir}/{r}.mp3"
        # text_to_speech(text=speak_text, role=r, language='en', filename=wavname, tts_type='edgeTTS')
        convert_to_wav(wavname, f"{homedir}/wav/{r}.wav")

def parse_srt_time(time_str):
    """Convert SRT time format to seconds."""
    h, m, s, ms = re.split('[:,]', time_str)
    return int(h) * 3600 + int(m) * 60 + float(s) + float(ms) / 1000

# 从字幕文件中，提取音频片段，用于做embedding检测
def extrace_audio_from_srt(input_wav, srt_file, output_dir):
    with open(srt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if(os.path.exists(output_dir) == False):
        os.makedirs(output_dir)

    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit(): 
            subtitle_index = lines[i].strip()
            subtitle_index_padded = subtitle_index.zfill(2)
            start_time = parse_srt_time(lines[i+1].split(' --> ')[0])
            end_time = parse_srt_time(lines[i+1].split(' --> ')[1])
            # 按行号命名
            output_file = f"{output_dir}/st_{subtitle_index_padded}.wav"

            subprocess.run([
                "ffmpeg",
                "-i", input_wav,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-ac", "1",
                "-ar", "16000",
                output_file
            ])
        i += 2

def get_debug_wav_files(wav_dir):
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    # 创建一个字典，键是角色名称，值是文件路径
    role_wav_dict = {}
    for wav_file in wav_files:
        # 角色名称是从文件名字符中提取
        role_name = wav_file.replace('.wav', '')
        full_path = os.path.join(wav_dir, wav_file)
        
        role_wav_dict[role_name] = full_path
    
    return role_wav_dict

def get_vec_embedding(audio_file):
    audio , _ =librosa.load(audio_file, sr=16000)    
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        output = model(input_values).last_hidden_state.mean(dim=1)

        return output[0].numpy()
    
def get_vec_similar(audio_file1, audio_file2):
    return 1- cosine(get_vec_embedding(audio_file1), get_vec_embedding(audio_file2))

# 直接拼接
def concact_embedding(embedding1, embedding2):
    return np.concatenate((embedding1, embedding2), axis=0)

# 加权平均
def concact_embedding_weight_average(embedding1, embedding2):
    # 检查嵌入的长度
    len_A = len(embedding1)
    len_B = len(embedding2)

    if len_A > len_B:
        # 将 B 扩展到与 A 相同的维度
        B_expanded = np.pad(embedding2, (0, len_A - len_B), 'constant')
        A_expanded = embedding1
    elif len_B > len_A:
        # 将 A 扩展到与 B 相同的维度
        A_expanded = np.pad(embedding1, (0, len_B - len_A), 'constant')
        B_expanded = embedding2
    else:
        # 如果长度相同，不需要扩展
        A_expanded = embedding1
        B_expanded = embedding2

    # 定义权重
    alpha = 0.6
    beta = 0.4

    # 计算加权平均
    weighted_average = alpha * A_expanded + beta * B_expanded
    return weighted_average

# 双塔模型
class DualTowerModel(nn.Module):
    def __init__(self, pyannote_dim, wav2vec_dim, hidden_dim):
        super(DualTowerModel, self).__init__()
        self.pyannote_tower = nn.Sequential(
            nn.Linear(pyannote_dim, hidden_dim),
            nn.ReLU()
        )
        self.wav2vec_tower = nn.Sequential(
            nn.Linear(wav2vec_dim, hidden_dim),
            nn.ReLU()
        )
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, pyannote_features, wav2vec_features):
        pyannote_embedding = self.pyannote_tower(pyannote_features)
        wav2vec_embedding = self.wav2vec_tower(wav2vec_features)
        combined_embedding = torch.cat([pyannote_embedding, wav2vec_embedding], dim=-1)
        fused_embedding = self.fusion_layer(combined_embedding)
        return fused_embedding

# 双塔模型融合
def concact_embedding_dual_tower(embedding1, embedding2):
    model = DualTowerModel(pyannote_dim=embedding1.shape[0], wav2vec_dim=embedding2.shape[0], hidden_dim=256)
    tensor_embedding1 = embedding1
    tensor_embedding2 = embedding2
    if isinstance(embedding1, np.ndarray):
        tensor_embedding1 = torch.from_numpy(embedding1)
    if isinstance(embedding2, np.ndarray):
        tensor_embedding2 = torch.from_numpy(embedding2)
    return model(tensor_embedding1, tensor_embedding2).detach().numpy()

# def cluster_embedding(embedding_list, n_clusters=None):
#     cluster = sklearn.cluster
#     labels = cluster.KMeans(n_clusters=n_clusters).fit_predict(embedding_list)

#     # # 使用层次聚类
#     clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0)
#     labels = clustering.fit_predict(features)
from datetime import timedelta
def parse_srt_time_for_segment(time_str):
    hours, minutes, seconds_milliseconds = time_str.split(':')
    seconds, milliseconds = seconds_milliseconds.split(',')
    return timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds), milliseconds=int(milliseconds))

# 从字幕文件中提取时间信息，组成segment列表并返回
def extract_segments_from_srt(srt_file):
    segments = []
    with open(srt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            start_time = parse_srt_time(lines[i+1].split(' --> ')[0])
            end_time = parse_srt_time(lines[i+1].split(' --> ')[1])
            segments.append((start_time, end_time))
        i += 2
    
    return segments

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
def cluster_segments(segments, audio_file):
    # 创建一个 Annotation 对象
    annotation = Annotation()
    for start, end in segments:
        annotation[Segment(start, end)] = 'unknown'

    # 提取每个分离段落的嵌入特征
    features = []
    for segment in annotation.itersegments():
        embedding = get_embedding(embedding_model, audio_file, segment)
        # embedding = get_embedding_for_vec(audio_file, segment)        
        features.append(embedding)

    # 将特征转换为numpy数组
    features = np.array(features)

    # 标准化特征数据
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 使用K-means进行聚类
    n_clusters = 3  # 预期有几个说话人
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # labels = kmeans.fit_predict(features)
    # labels = kmeans.fit_predict(features_scaled)

    # 计算余弦距离矩阵
    # cosine_distances = pdist(features_scaled, metric='cosine')

    # # 将距离矩阵转换为方阵形式
    # cosine_distances_square = squareform(cosine_distances)

    # # 计算层次聚类的链接矩阵
    # Z = linkage(cosine_distances_square, method='ward')

    # # # 绘制树状图
    # plt.figure(figsize=(10, 5))
    # dendrogram(Z)
    # plt.title('Dendrogram')
    # plt.xlabel('Sample index')
    # plt.ylabel('Distance')
    # plt.show()

    clustering = AgglomerativeClustering(n_clusters, affinity='cosine', linkage='average')
    labels = clustering.fit_predict(features_scaled)

    # 将聚类结果添加到 Annotation 对象中
    for segment, label in zip(annotation.itersegments(), labels):
        annotation[segment] = f'speaker_{label}'

    # 打印聚类结果
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        print(f"Segment {segment.start:.1f}s to {segment.end:.1f}s is spoken by {speaker}")

if __name__ == '__main__':
    dir_name = '301'
    ##### 根据字幕文件中的时间片定义，提取出16k、单声道规格的音频片段 #####
    # input_wav = f"F:\\Project\\test\\{dir_name}\\vocal.wav"
    # srt_file = f"F:\\Project\\test\\{dir_name}\\zh-cn.srt"
    # output_dir = f"F:\\Project\\test\\{dir_name}\\debug"
    # extrace_audio_from_srt(input_wav, srt_file, output_dir)
    ########################################

    ##### 生成edgetts所有角色的配音文件##### 
    # make_role_audio()
    # fl = get_role_wav_files()
    # print(fl)
    ######################################## 

    ##### 提取所有角色配音文件的特征值，并保存到文件 #####
    # extract_embeddings(save_embeddings)
    ########################################

    ##### 读取embedding ####
    # audio_file = "F:\\Project\\test\\101\\debug\\st_01.wav"    
    # audio_file1 = "F:\\Project\\test\\101\\debug\\subtitle_1.wav"    
    # embedding = get_embedding_from_audio(audio_file)
    # embedding = get_voiced_embedding_from_audio(audio_file)
    # embedding = get_vec_embedding(audio_file)
    # embedding = concact_embedding(get_embedding_from_audio(audio_file), get_vec_embedding(audio_file))
    # print(embedding)
    ##########################
    
    # role, similarity = find_from_roles(embedding)
    # print(f"The most similar role is {role} with similarity {similarity[role]}")

    ##### 字幕片段，和所有配音角色比对相似度 ##### 
    # audio_files = get_debug_wav_files(output_dir)
    # role_embeddings = get_role_wav_embeddings()
    # for title, audio_file in audio_files.items():
    #     embedding = concact_embedding(get_embedding_from_audio(audio_file), get_vec_embedding(audio_file))
    #     # role, similarity = find_from_roles(embedding)
    #     role, similarity = find_most_similar_role(embedding, role_embeddings)
    #     print(f"{title}: The most similar role is {role} with similarity {similarity[role]}")
    ###############################################

    ##### 俩音频文件比对相似度 ####
    # similarity = get_similar(f"{homedir}/wav/en-US-AnaNeural.wav", audio_file)
    # similarity = get_similar("F:\\Project\\test\\101\\06.wav", audio_file)
    # print(similarity)
    ##########################
    
    ##### 取目录下所有wav文件，计算特征值，并依次和其他wav文件的特征值进行相似度检测 ####
    # output_dir = f"F:\\Project\\test\\{dir_name}\\debug"
    # audio_files = get_debug_wav_files(output_dir)
    # # 创建一个字典来存储每个角色的embedding
    # embeddings = {}
    # # 遍历角色和音频文件，计算并存储embedding
    # for role, audio_file in audio_files.items():
    #     embeddings[role] = get_embedding_from_audio(audio_file)
    #     # embeddings[role] = get_vec_embedding(audio_file)
    #     # embeddings[role] = concact_embedding(get_embedding_from_audio(audio_file), get_vec_embedding(audio_file))
    #     # embeddings[role] = concact_embedding_weight_average(get_embedding_from_audio(audio_file), get_vec_embedding(audio_file))
    #     # embeddings[role] = concact_embedding_dual_tower(get_embedding_from_audio(audio_file), get_vec_embedding(audio_file))
        
    # similarity_df = pd.DataFrame(index=audio_files.keys(), columns=audio_files.keys())
    # for role1, audio_file in audio_files.items():
    #     for role2, audio_file1 in audio_files.items():
    #         # if role2 == role1:
    #         #     continue
    #         similarity = calculate_similarity(embeddings[role1], embeddings[role2])            
    #         similarity_df.at[role1, role2] = similarity
    
    # print(similarity_df)
    ##############################################################################

    #################### 测试音频片段的聚类 ###################
    srt_file = f"F:\\Project\\test\\{dir_name}\\zh-cn.srt"
    audio_file = f"F:\\Project\\test\\{dir_name}\\vocal.wav"
    segments = extract_segments_from_srt(srt_file)
    cluster_segments(segments, audio_file)
    ######################################