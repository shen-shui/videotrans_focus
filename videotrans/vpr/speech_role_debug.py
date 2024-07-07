import os
import subprocess

from videotrans.tts import text_to_speech
from videotrans.util import tools
from pathlib import Path
import numpy as np
import librosa

from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment
from pyannote.audio.pipelines import VoiceActivityDetection

from scipy.spatial.distance import cosine

homepath=Path.home()/'Videos/pyvideotrans/tts'
homedir = homepath.as_posix()
wav_dir = os.path.join(homedir, 'wav')
embedding_path=os.path.join(wav_dir,"role_embeddings.npy")

if not os.path.exists(homedir):
    os.makedirs(homedir, exist_ok=True)
if not os.path.exists(wav_dir):
    os.makedirs(wav_dir, exist_ok=True)

def extract_embedding(audio_file, embedding_model):
    y, sr = librosa.load(audio_file, sr=None)
    embedding = embedding_model({"waveform": y, "sample_rate": sr})
    return embedding

# 提取音频片段的embedding
def get_embedding(embedding_model, audio_path, segment):
    inference = Inference(embedding_model, window="whole")
    embedding = inference.crop(audio_path, segment)

    return embedding

# 提取音频的segment
def extract_audio_segment(audio_file):
    y, sr= librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    # 取整个语音片段
    segment = Segment(0, duration)

    return segment

def save_embeddings(role_audio_files, embedding_model, save_path=embedding_path):
    embeddings = {}
    for role, audio_file in role_audio_files.items():
        segment = extract_audio_segment(audio_file)
        embedding = get_embedding(embedding_model, audio_file, segment)
        embeddings[role] = embedding
    np.save(save_path, embeddings)

# 提取并保存配音角色的特征值
def extract_embeddings(save_embeddings):
    embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token='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')
    save_embeddings(get_role_wav_files(), embedding_model)

def get_role_mp3_files(dir = homedir):
    mp3_files = [f for f in os.listdir(dir) if f.endswith('.mp3')]
    return mp3_files

#  
#   返回格式如下：
#   {
#      "NeerjaNeural": "path/to/en-IN-NeerjaNeural.wav",
#      "MitchellNeural": "path/to/en-NZ-MitchellNeural.wav"
#   }
#  
def get_role_wav_files():
    # 获取所有以.wav结尾的文件
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    # 创建一个字典，键是角色名称，值是文件路径
    role_wav_dict = {}
    for wav_file in wav_files:
        # 角色名称是从文件名字符中提取
        role_name = wav_file.split('-')[-1].replace('.wav', '')
        full_path = os.path.join(wav_dir, wav_file)
        
        # 如果角色名称已经存在于字典中，就将其添加到现有列表中
        # if role_name in role_wav_dict:
        #     role_wav_dict[role_name].append(full_path)
        # else:
        #     role_wav_dict[role_name] = [full_path]
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
    most_similar_role = max(similarities, key=similarities.get)
    return most_similar_role, similarities

# 匹配一个最合适的角色
def get_role(audio_embedding):
    # 加载预先保存的配音角色特征值
    saved_embeddings = np.load(embedding_path, allow_pickle=True).item()
    role, similarity = find_most_similar_role(audio_embedding, saved_embeddings)

    print(f"The most similar role is {role} with similarity {similarity}")
    return role

if __name__ == '__main__':
    role_list = tools.get_edge_rolelist()
    for r in list(role_list['en']):
        if r == 'No':
            continue
        wavname = f"{homedir}/{r}.mp3"
        text_to_speech(text="Hello, my dear friend. I hope your every day is beautiful and enjoyable!", role=r, language='en', filename=wavname, tts_type='edgeTTS')
        convert_to_wav(wavname, f"{homedir}/wav/{r}.wav")
        
    fl = get_role_wav_files()
    print(fl)
    extract_embeddings(save_embeddings)
    


