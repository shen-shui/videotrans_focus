import os
import subprocess

import torch
import torchaudio

from videotrans.tts import text_to_speech
from videotrans.util import tools
from pathlib import Path
import numpy as np
import librosa

from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment

from scipy.spatial.distance import cosine

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
        embedding = get_embedding_from_audio(audio_file)
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
        '-ar', '44100',  # 设置采样率为44100Hz
        '-vn',  # 忽略视频流
        '-c:a', 'pcm_s16le',  # 设置音频编码为16位线性PCM
        out_path
    ]
    subprocess.run(command, check=True)

def calculate_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)  # 相似度值，越接近1越相似

def find_most_similar_role(new_audio_embedding, saved_embeddings):
    similarities = {}
    for role, embedding in saved_embeddings.items():
        similarity = calculate_similarity(new_audio_embedding, embedding)
        similarities[role] = similarity
        print(f"Similarity between {role} and new audio: {similarity}")
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
    segment = extract_audio_segment(audio_file)
    embedding = get_embedding(embedding_model, audio_file, segment)
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
        audio_embedding = get_embedding_from_audio(audio_file)
        # 取音频中有声音的部分，计算特征值，效果待验证
        # embedding = get_voiced_embedding_from_audio(audio_file)

        similarity = calculate_similarity(embedding, audio_embedding)
        similarities[role] = similarity
        print(f"Similarity between {role} and new audio: {similarity}")
    most_similar_role = max(similarities, key=similarities.get)
    return most_similar_role, similarities

# 俩音频的相似度值
def get_similar(audio_file1, audio_file2):

    similarity = calculate_similarity(get_embedding_from_audio(audio_file1), get_embedding_from_audio(audio_file2))    
    
    # similarity = calculate_similarity(get_voiced_embedding_from_audio(audio_file1), get_voiced_embedding_from_audio(audio_file2))
    return similarity

# 生成所有edge的角色配音文件，方便调试
def make_role_audio():
    speak_text = 'Hello, my dear friend. I hope your every day is beautiful and enjoyable!'
    role_list = tools.get_edge_rolelist()
    for r in list(role_list['en']):
        if r == 'No':
            continue
        wavname = f"{homedir}/{r}.mp3"
        text_to_speech(text=speak_text, role=r, language='en', filename=wavname, tts_type='edgeTTS')
        convert_to_wav(wavname, f"{homedir}/wav/{r}.wav")

if __name__ == '__main__':

    # make_role_audio()
        
    # fl = get_role_wav_files()
    # print(fl)

    # extract_embeddings(save_embeddings)

    audio_file = "F:\\Project\\test\\101\\06.wav"    
    embedding = get_voiced_embedding_from_audio(audio_file)
    
    role, similarity = find_from_roles(embedding)
    print(f"The most similar role is {role} with similarity {similarity[role]}")

    # similarity = get_similar(f"{homedir}/wav/en-US-AnaNeural.wav", audio_file)
    # similarity = get_similar(f"{homedir}/wav/en-US-RogerNeural.wav", audio_file)
    # print(similarity)
    


