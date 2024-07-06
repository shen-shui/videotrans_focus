import os
import librosa
import numpy as np

from pyannote.audio import Pipeline
from pyannote.core import Segment
from pyannote.core import notebook

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import random
import pickle

from videotrans.configure import config
from videotrans.util import tools

# 指定下载目录
# cache_dir = "F:\\huggingface_cache"

# 使用cached_path确保文件下载到指定目录
# config_url = "https://huggingface.co/pyannote/speaker-diarization-3.0/resolve/main/config.yaml"
# config_path = cached_path(config_url, cache_dir=download_dir)

# 指定超参数文件路径
hparams_path = "F:\\gitwork-chroya\\videotrans_focus\\videotrans\\vpr\\hparam.yaml"

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token='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')
result = [{
        'start_time': 0,
        'end_time': 0,
        'duration': 0,
        'speaker_id': 0,
        'gender': 1, # 1:男，0:女
        'speaker_name': 'unknown'}]

# 加载音频文件
audio_file = "F:\\Project\\test\\11\\vocal.wav"

def get_speaker_result(audio_file, output_dir=os.getcwd()):
    # 运行说话人分割和识别
    diarization = pipeline(audio_file)
    speakers = set()
    result = []
    MIN_DURATION_ON = 0.1

    # 导出为RTTM格式
    with open(get_plot_path("speaker.rttm"), "w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            record = {}
            start_time = turn.start
            duration = turn.duration

            if(duration < MIN_DURATION_ON) :
                # 100毫秒以内的，忽略
                continue

            segment = Segment(turn.start, turn.end)
            gender = get_gender_from_segment(audio_file, segment)

            end_time = start_time+ duration
            speakers.add(speaker)
            
            start_time_milliseconds = int(start_time * 1000)
            end_time_milliseconds = int(end_time * 1000)
            record['start_time'] = start_time_milliseconds
            record['end_time'] = end_time_milliseconds
            record['duration'] = duration
            record['speaker_id'] = speaker
            record['gender'] = gender
            result.append(record)

            data = f"SPEAKER 1 {start_time:.3f} {duration:.3f} {gender} {speaker}\n"
            print(data)
            f.write(data)
    
    tools.set_process(f"识别到说话人数量：{len(speakers)}")

    show_speaker_plot(diarization, output_dir)
    
    return result

def get_speaker_count(speaker_result):
    return len(set(item['speaker_id'] for item in speaker_result))

# 说话人识别的信息图形化展示
def show_speaker_plot(diarization, output_dir):
    # 绘制
    fig = notebook.plot_annotation(diarization, time=True)
    plt.savefig(get_plot_path("speaker_plot.png"))

# 都存到plot目录中
def get_plot_path(file_name):
    if not os.path.exists(os.getcwd()+'/plot_debug'):
        os.makedirs(os.getcwd()+'/plot_debug')
    
    return os.path.join(os.getcwd(), 'plot_debug', file_name)

# 把字幕文件图形化展示
def show_subtitle_plot(sub_list, output_dir=os.getcwd()):
    # 解析数据，准备绘图
    times = []
    durations = []
    texts = []
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # 颜色列表，可扩展

    for i, it in enumerate(sub_list):
        times.append(it['start_time'])  # 假设时间是以秒为单位的元组，这里取开始秒数
        durations.append(it['end_time'] - it['start_time'])
        texts.append(it['line'])

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 2))

    for i, (time, duration, text) in enumerate(zip(times, durations, texts)):
        ax.add_patch(mpatches.Rectangle((time, 0), duration, 1, facecolor=colors[i % len(colors)]))
        # 添加标签
        ax.text(time + duration / 2, 0.8, text, ha='center', va='center')

    ax.set_yticks([])  # 隐藏y轴刻度
    ax.set_xlabel('Time (seconds)')
    plt.title('Subtitle Visualization')
    plt.xlim(left=min(times), right=max(times+durations))
    # plt.show()
    plt.savefig(get_plot_path("subtitle_plot.png"))

def show_line_role_plot(line_roles, output_dir=os.getcwd()):
    # 解析数据，准备绘图
    times = []
    durations = []
    texts = []
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # 颜色列表
    speaker_set = set()

    for key in line_roles.keys():
        if(not isinstance(key, str)):
            continue
        it = line_roles[key]
        times.append(it['start_time'])  # 取开始秒数
        durations.append(it['end_time'] - it['start_time'])
        texts.append(it['speaker_id'])
        speaker_set.add(it['speaker_id'])
    
    speaker_list = list(speaker_set)

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 2))

    for i, (time, duration, text) in enumerate(zip(times, durations, texts)):
        # 在list中寻找speaker_id的位置
        spk_index = speaker_list.index(text)
        spk_index = spk_index if spk_index >= 0 else 0
        ax.add_patch(mpatches.Rectangle((time, 0), duration, 1, facecolor=colors[spk_index]))
        # 添加标签
        ax.text(time + duration / 2, 0.8, text.split('_')[-1], ha='center', va='center')

    ax.set_yticks([])  # 隐藏y轴刻度
    ax.set_xlabel('Time (seconds)')
    plt.title('Linerole Visualization')
    plt.xlim(left=min(times), right=max(times+durations))
    # plt.show()
    plt.savefig(get_plot_path("line_role_plot.png"))

# 定义一个函数来提取基频并判断性别
def get_gender_from_segment(audio_file, segment):
    y, sr = librosa.load(audio_file, sr=None, offset=segment.start, duration=segment.duration)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    mean_f0 = np.mean(f0[voiced_flag])
    
    male_threshold = 160
    
    if mean_f0 > male_threshold:
        return 0
    
    return 1

def aggregate_by_speaker(result):
    if not result:  # 如果列表为空，直接返回
        return []

    aggregated_result = []
    current_group = result[0]  # 初始化当前聚合组为第一个元素
    line = 1 #字幕的行从1开始，这里也从1开始

    tools.set_process(f"说话人聚合")

    for item in result[1:]:
        # 如果当前项的speaker_id与聚合组的不同，则结束当前组并开始新的组
        if item['speaker_id'] != current_group['speaker_id']:
            aggregated_result.append(current_group)  # 将当前组添加到结果列表

            # 开始新的聚合组
            current_group = item
            # tools.set_process(f"{line} {current_group['speaker_id']} :{current_group['start_time']}->{current_group['end_time']}")
            line += 1
        else:  # 如果speaker_id相同，则更新当前组的end_time
            # 更新end_time
            current_group['end_time'] = item['end_time']
            # 累加duration，只是说话的duration，期间的空白没有计入
            current_group['duration'] += item['duration']

    # 添加最后一个聚合组到结果列表
    aggregated_result.append(current_group)

    for index, group in enumerate(aggregated_result):        
        tools.set_process(f"{index+1} {group['speaker_id']} :{group['start_time']}->{group['end_time']}")

    return aggregated_result

def define_line_roles(sub_list, speaker_result, role_list, default_role=None):
    # 两个参数都不能为空
    if sub_list is None or speaker_result is None or not role_list:
        return
    
    # line_roles定义为一个字典    
    line_roles = {}
    # 最多允许的间隔时间
    max_gap = 200 

    # 先聚合一下
    speaker_result = aggregate_by_speaker(speaker_result)
    
    # 获取所有不同的说话人ID
    unique_speakers = set(item['speaker_id'] for item in speaker_result)
    
    # 创建一个说话人ID到角色的随机映射
    speaker_to_role_mapping = {}
    for speaker in unique_speakers:
        # 从role_list中随机选择一个角色分配给说话人
        while True:
            role = random.choice(list(role_list['en']))
            if role != "No":
                break
        speaker_to_role_mapping[speaker] = fit_edge_role(role, default_role)
    
    # speaker_result列表的索引
    speaker_index = 0
    tools.set_process("字幕匹配说话人")
    for it in sub_list:
        # 要么用当前匹配上的角色，要么用上一个匹配上的角色
        sub_start_time,sub_end_time,line = it['start_time'], it['end_time'], it['line']

        speaker_obj = speaker_result[speaker_index]
        speaker_start_time, speaker_end_time, speaker_id = speaker_obj['start_time'], speaker_obj['end_time'], speaker_obj['speaker_id']

        # 当前的字幕匹配角色
        speaker_id = speaker_result[speaker_index]['speaker_id']
        role = speaker_to_role_mapping.get(speaker_id, default_role)
        line_roles[line] = role

        # 额外信息，用来showplot
        line_role_obj = {}
        line_role_obj['role'] = role
        line_role_obj['speaker_id'] = speaker_id
        line_role_obj['start_time'] = sub_start_time
        line_role_obj['end_time'] = sub_end_time 
        line_roles[str(line)] = line_role_obj

        tools.set_process(f"{line} {role} :{sub_start_time}->{sub_end_time}")

        if(sub_end_time > speaker_end_time - max_gap):
            # 需要移动索引的情况
            speaker_index += 1
        
        # if(speaker_start_time - max_gap <= sub_start_time <= sub_end_time <= speaker_end_time + max_gap):
        #     # 字幕在配音时间范围内
        #     continue
        # elif sub_end_time > speaker_end_time + max_gap:
        #     # end还在范围内
        #     continue
        # else: 
        #     # 超出范围了，移动索引，看下一个
        #     speaker_index += 1


        # if sub_start_time - speaker_start_time <= max_gap or speaker_start_time - sub_start_time <= max_gap :
        #     # 起点接近，成功匹配，speaker移动到下一个
        #     speaker_index += 1 
        # else:
        #     # 说明说话人的起点不在当前字幕的附近，那就看终点
        #     if speaker_end_time - sub_end_time <= max_gap :
        #         # 终点没超过字幕太多，成功匹配，speaker移动到下一个
        #         speaker_index += 1
        #     else:
        #         # 终点超过字幕太多，说明这个说话人没有匹配上，那就跳过
        #         continue
    
    return line_roles
    
def fit_edge_role(role_item, default_role):
    return role_item if role_item != "No" else default_role
    # return role_item[1] if role_item is not None else default_role

if __name__ == '__main__':
    config.params['detail_log'] = True
    
    sr = get_speaker_result(audio_file)

    with open('subs_data.pickle', 'rb') as handle:
        subs = pickle.load(handle)

    role_list = tools.get_edge_rolelist()

    line_roles = define_line_roles(subs, sr, role_list, "default")

    show_subtitle_plot(subs)
    show_line_role_plot(line_roles)
