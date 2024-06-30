import librosa
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment
import random

from videotrans.util import tools

# 指定下载目录
# cache_dir = "F:\\huggingface_cache"

# 使用cached_path确保文件下载到指定目录
# config_url = "https://huggingface.co/pyannote/speaker-diarization-3.0/resolve/main/config.yaml"
# config_path = cached_path(config_url, cache_dir=download_dir)

# 指定超参数文件路径
hparams_path = "F:\\gitwork-chroya\\videotrans_focus\\videotrans\\vpr\\hparam.yaml"

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token='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')
result = [{
        'start_time': 0,
        'end_time': 0,
        'duration': 0,
        'speaker_id': 'speaker_00',
        'gender': 1, # 1:男，0:女
        'speaker_name': 'unknown'}]

# 加载音频文件
audio_file = "F:\\Project\\aigc短剧\\第一集1分钟\\第一集1分钟\\vocal.wav"

def get_speaker_result(audio_file, output_dir):
    # 运行说话人分割和识别
    diarization = pipeline(audio_file)
    speakers = set()
    result = []
    MIN_DURATION_ON = 0.1

    # 导出为RTTM格式
    with open(output_dir+"/speaker.rttm", "w") as f:
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

            f.write(f"SPEAKER 1 {start_time:.3f} {duration:.3f} {gender} {speaker}\n")
    
    tools.set_process(f"识别到说话人数量：{len(speakers)}")
    
    return result

def get_speaker_count(speaker_result):
    return len(set(item['speaker_id'] for item in speaker_result))

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

        if(sub_end_time > speaker_end_time):
            # 需要移动索引的情况
            speaker_index += 1
        
        # 当前的字幕匹配角色
        speaker_id = speaker_result[speaker_index]['speaker_id']
        role = speaker_to_role_mapping.get(speaker_id, default_role)
        line_roles[line] = role

        tools.set_process(f"{line} {role} :{sub_start_time}->{sub_end_time}")

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
    get_speaker_result(audio_file)
    print(pipeline)