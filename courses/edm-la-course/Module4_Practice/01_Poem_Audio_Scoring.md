## 习题一：一年级古诗朗读音频分析与打分

**背景：** 老师提供了200个一年级同学朗读《风》（唐·李峤）的音频文件。
**诗歌内容：** "解落三秋月，能开二月花。过江千尺浪，入竹万竿斜。" (共20个字，4句，每句5字)
**要求：**
1.  用 Python 抽取这些音频文件的结构化特征（至少5个）。
2.  根据这些结构化特征，给这个音频文件进行朗读打分。

---

### 第一部分：理清思路 —— 我们要怎么做？

在开始动手之前，我们先一起思考一下整个任务的流程和关键点：

1.  **理解任务目标：**
    * **核心：** 对朗读音频进行**量化评估**。
    * **关键：** “打分”依据什么？对于一年级的小朋友朗读这首20字的五言绝句，什么样的朗读算是“好”的？我们提取的“结构化特征”需要能反映这些“好”的方面。

2.  **思考“好”的朗读可能包含哪些元素（尤其针对低年级）：**
    * **流畅度：** 是否能顺利读下来，中间不合适的停顿是否较少？
    * **完整性：** 是否读完了所有字？（这个任务可能假设都读完了，但真实场景需要考虑）
    * **适当的停顿：** 五言诗通常在每句末尾有自然停顿。能否在“月、花、浪、斜”后有合理的停顿？
    * **语速：** 是否过快或过慢？
    * **音量：** 声音是否清晰可闻，大小适中？
    * **（进阶）富有感情/韵律感：** 这点对于一年级可能要求不高，且较难从纯结构化特征判断，但可以作为思考方向。比如音高(Pitch)的变化，音强的起伏。

3.  **从“朗读好坏的元素”到“可抽取的结构化特征”：**
    * **流畅度/停顿：**
        * 音频总时长。
        * 纯语音时长 vs. 静音时长。
        * 静音片段的数量和平均时长（可以区分长、短静音）。
        * 尤其关注是否在句末（第5, 10, 15, 20字后）有停顿。
    * **语速：**
        * （总字数 / 纯语音时长）得到的平均语速。
    * **音量：**
        * 平均音量/音强 (RMS energy)。
        * 音量变化范围（最大最小音量差，或标准差）。
    * **（进阶）韵律感：**
        * 基频（F0 / Pitch）的平均值和标准差（反映音高和语调变化）。

4.  **选择合适的Python工具库：**
    * **`librosa`：** 非常强大的音频分析库，可以提取音高、音强、节拍、MFCC等多种特征。
    * **`pydub`：** 对于音频文件的基本操作（如加载、切分、时长计算、静音检测）非常方便。
    * **`wave` / `soundfile`：** 用于读写特定格式的音频文件（如 .wav）。
    * **`numpy` / `pandas`：** 用于数值计算和数据组织。
    * **`os`：** 用于遍历文件。

5.  **设计打分逻辑：**
    * **基于规则：** 为每个提取的特征设定一个“理想范围”或“阈值”。例如：
        * 总时长在10-15秒之间得XX分。
        * 句末停顿清晰，每次停顿在0.5-1.5秒之间得XX分。
        * 意外的长停顿（>2秒）少于N次得XX分。
        * 平均音量在某个分贝范围得XX分。
    * **加权评分：** 给不同的特征或规则赋予不同的权重，然后综合计算总分。
    * **（更高级）机器学习模型：** 如果我们有一批已经由人工打好分的音频作为“标准答案”（训练数据），理论上可以训练一个回归模型来预测分数。但这超出了本习题“至少5个结构化特征并打分”的基本要求，可以作为未来探索。

6.  **实施步骤：**
    * 遍历所有音频文件。
    * 对每个文件加载并提取特征。
    * 将特征存起来（比如一个列表，或者Pandas DataFrame）。
    * 应用打分逻辑计算每个文件的分数。
    * 分析打分结果。

7.  **挑战与思考：**
    * **静音检测的参数调整：** `min_silence_len` (最小静音长度) 和 `silence_thresh` (静音阈值) 对结果影响很大，需要根据实际音频情况调整。
    * **句末停顿的准确识别：** 对于一年级学生，朗读节奏可能不标准，准确识别“句末”停顿有挑战。可能需要结合总时长和预期句长来估计。
    * **“好”的主观性：** 即使是结构化特征，如何组合它们来反映朗读的“好坏”，也带有一定主观性，需要不断调试和验证。

---

### 第二部分：小步子任务单 —— 完成作业的推荐步骤

1.  **环境搭建与熟悉：**
    * 确保你的 Python 环境已安装。
    * 安装必要的库：`pip install librosa pydub numpy pandas matplotlib` (matplotlib用于可选的可视化)。
    * 尝试加载一个音频文件，播放它，看看它的波形图，熟悉基本操作。

2.  **单文件特征提取实践：**
    * 选择**一个**音频文件作为试点。
    * **任务2.1：提取总时长。** (使用 `pydub` 或 `librosa`)
    * **任务2.2：提取平均音量/强度。** (使用 `librosa.feature.rms` 计算均方根能量)
    * **任务2.3：检测静音片段，计算总语音时长、总静音时长、静音次数、平均静音时长。** (重点使用 `pydub.silence.detect_nonsilent_ranges` 或 `split_on_silence`，需要仔细调整参数)。
    * **任务2.4：计算平均语速。** (诗歌总字数 / 总语音时长)。
    * **任务2.5：句末停顿分析（简化版）。**
        * 诗歌共4句，理想情况下应有3-4个主要停顿。
        * 尝试统计时长在某个合理范围（比如0.5秒-2秒）的停顿数量。
        * （进阶思考）能否大致判断这些主要停顿是否发生在句末？（例如，基于总时长的平均分配来估计每句结束的时间点，看是否有停顿落在附近。这比较粗略，但可以尝试。）
    * **（可选）任务2.6：提取平均音高及其标准差。** (使用 `librosa.pyin`，注意处理 `NaN` 值)。

3.  **批量处理所有音频文件：**
    * 编写一个循环，遍历200个音频文件。
    * 对每个文件执行步骤2中的特征提取过程。
    * 将每个文件的提取到的特征（文件名、总时长、平均音量、静音次数、平均语速等）存储在一个结构化的列表中，每项是一个字典，或者直接构建 Pandas DataFrame。

4.  **设计并实现打分函数：**
    * 定义一个函数，输入是一个音频文件的特征集合（比如一个字典或DataFrame的一行）。
    * 在函数内部，根据你对“好”的朗读的理解，基于这些特征设计一套评分规则。
        * 例如：时长在10-15秒得20分，15-20秒得15分等。
        * 句末停顿（假设我们能识别出3-4个主要停顿）得20分，少于3个或多于5个酌情扣分。
        * 平均语速在1.5-2.5字/秒（纯语音时长）得20分。
        * 平均音量适中（需要根据实际数据分布确定范围）得20分。
        * 额外的不必要长停顿（如超过2秒的非句末停顿）每出现一次扣X分。
    * 函数输出该音频的总分（例如0-100分）。

5.  **对所有文件打分并分析结果：**
    * 将打分函数应用到所有音频文件的特征数据上。
    * 查看分数的分布（最高分、最低分、平均分、直方图）。
    * 找出得分最高和最低的几个音频，听一听，看看是否与你的打分逻辑预期一致。如果不一致，反思并调整打分规则或特征提取方法。

6.  **撰写报告/总结：**
    * 描述你提取了哪些特征，为什么选择它们。
    * 详细说明你的打分逻辑和权重（如果有的话）。
    * 展示你的打分结果，并进行简要分析。
    * 讨论你的方法的优点、局限性以及未来可以改进的方向。

---

### 第三部分：参考答案（思路与关键Python代码片段提示）

**重要声明：** 以下代码片段仅为示例和功能提示，**并非完整可直接运行的解决方案**。你需要根据实际音频文件的特点（如格式、音量大小、背景噪音等）仔细调整参数，并补全文件处理、循环、数据存储等逻辑。

```python
import librosa
import librosa.display # 用于可视化（可选）
from pydub import AudioSegment
from pydub.silence import detect_nonsilent_ranges, split_on_silence # 用于静音检测
import numpy as np
import pandas as pd
import os # 用于文件操作

# 诗歌信息
POEM_TEXT = "解落三秋月能开二月花过江千尺浪入竹万竿斜"
NUM_CHARS = len(POEM_TEXT) # 20 字
NUM_LINES = 4

def extract_features_from_audio(audio_path):
    features = {}
    features['filename'] = os.path.basename(audio_path)

    try:
        # 使用 librosa 加载音频 (主要用于频谱相关特征)
        y, sr = librosa.load(audio_path, sr=None) # sr=None 保留原始采样率
        
        # 1. 总时长 (librosa)
        features['total_duration_librosa_s'] = librosa.get_duration(y=y, sr=sr)

        # 2. 平均音量/强度 (RMS energy)
        rms_energy = librosa.feature.rms(y=y)[0] # 取[0]是因为rms返回的是一个二维数组
        features['avg_rms_energy'] = np.mean(rms_energy)
        features['std_rms_energy'] = np.std(rms_energy) # 音量变化

        # (可选) 3. 平均音高 (Pitch / F0) 及其标准差 (语调变化)
        # pyin 算法通常效果不错，但可能较慢
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        # 只取有声部分的音高
        pitch_values = f0[voiced_flag] 
        features['avg_pitch_hz'] = np.nanmean(pitch_values) if len(pitch_values[~np.isnan(pitch_values)]) > 0 else 0
        features['std_pitch_hz'] = np.nanstd(pitch_values) if len(pitch_values[~np.isnan(pitch_values)]) > 0 else 0

        # 使用 pydub 加载音频 (主要用于时长、静音等操作)
        # 假设是 wav 文件，根据实际情况修改
        try:
            audio_segment = AudioSegment.from_file(audio_path) # pydub 会自动识别格式
        except Exception as e_pydub_load:
            print(f"Pydub Error loading {audio_path}: {e_pydub_load}")
            # 如果pydub加载失败，一些基于它的特征会是默认值
            features['total_duration_pydub_s'] = features.get('total_duration_librosa_s', 0) #  fallback
            features['speech_duration_s'] = 0
            features['silence_duration_s'] = features.get('total_duration_pydub_s', 0)
            features['num_pauses'] = 0
            features['avg_pause_duration_s'] = 0
            features['speaking_rate_cps'] = 0
            features['speech_ratio'] = 0
            # 继续提取其他基于librosa的特征，或者直接返回已有的
            # return features # 或者只在pydub成功时继续

        # 1. 总时长 (pydub) - 更精确
        features['total_duration_pydub_s'] = audio_segment.duration_seconds

        # 4. 静音检测与语音时长分析 (pydub)
        # 需要根据实际音频调整 min_silence_len 和 silence_thresh
        # silence_thresh 通常是相对于音频的平均分贝或者最大分贝。audio_segment.dBFS 是平均分贝。
        # 例如，比平均分贝低 16 dBFS 的认为是静音
        silence_threshold = audio_segment.dBFS - 16 
        min_pause_len_ms = 400 # 认为超过400ms的静音算一个停顿 (可调整)
        
        # 获取非静音区间
        nonsilent_ranges = detect_nonsilent_ranges(
            audio_segment,
            min_silence_len=min_pause_len_ms, 
            silence_thresh=silence_threshold,
            seek_step=1
        )

        speech_duration_ms = 0
        for start_i, end_i in nonsilent_ranges:
            speech_duration_ms += (end_i - start_i)
        features['speech_duration_s'] = speech_duration_ms / 1000.0
        features['silence_duration_s'] = features['total_duration_pydub_s'] - features['speech_duration_s']
        features['speech_ratio'] = features['speech_duration_s'] / features['total_duration_pydub_s'] if features['total_duration_pydub_s'] > 0 else 0
        
        # 5. 停顿次数与平均停顿时间 (基于nonsilent_ranges推断)
        num_pauses = 0
        total_pause_duration_ms = 0
        
        # 句首是否有静音 (严格来说这不算pause, 但可以作为特征)
        # if nonsilent_ranges and nonsilent_ranges[0][0] > min_pause_len_ms:
        #     num_pauses +=1
        #     total_pause_duration_ms += nonsilent_ranges[0][0]

        if len(nonsilent_ranges) > 1:
            num_pauses = len(nonsilent_ranges) - 1 # 语音段之间的静音算作停顿
            for i in range(num_pauses):
                pause_start = nonsilent_ranges[i][1]
                pause_end = nonsilent_ranges[i+1][0]
                total_pause_duration_ms += (pause_end - pause_start)
        
        features['num_pauses'] = num_pauses # 这里指语音段之间的停顿
        features['avg_pause_duration_s'] = (total_pause_duration_ms / num_pauses / 1000.0) if num_pauses > 0 else 0

        # 6. 平均语速 (字/秒，基于纯语音时长)
        features['speaking_rate_cps'] = NUM_CHARS / features['speech_duration_s'] if features['speech_duration_s'] > 0 else 0

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # 为出错的文件填充默认值或NaN
        feature_keys = ['total_duration_librosa_s', 'avg_rms_energy', 'std_rms_energy', 
                        'avg_pitch_hz', 'std_pitch_hz', 'total_duration_pydub_s', 
                        'speech_duration_s', 'silence_duration_s', 'num_pauses', 
                        'avg_pause_duration_s', 'speaking_rate_cps', 'speech_ratio']
        for key in feature_keys:
            features.setdefault(key, np.nan) # 使用np.nan标记错误，方便后续处理
            
    return features

# --- 假设你有一个音频文件列表 audio_files ---
# all_audio_features = []
# audio_dir = "path/to/your/200_audio_files_directory/"
# for filename in os.listdir(audio_dir):
#     if filename.lower().endswith(('.wav', '.mp3')): # 根据你的文件格式调整
#         audio_path = os.path.join(audio_dir, filename)
#         current_features = extract_features_from_audio(audio_path)
#         all_audio_features.append(current_features)

# df_features = pd.DataFrame(all_audio_features)
# print(df_features.head())
# print(df_features.describe()) # 查看特征的统计分布，为打分规则设计提供参考

# --- 设计打分函数 (非常初步的示例) ---
def score_reading(row_features, poem_num_lines=NUM_LINES):
    score = 100 # 满分100
    penalties = []

    # 1. 总时长 (假设理想时长10-18秒)
    total_duration = row_features.get('total_duration_pydub_s', 0)
    if total_duration < 8 or total_duration > 20:
        penalties.append(f"时长不佳({total_duration:.1f}s): -15")
        score -= 15
    elif total_duration < 10 or total_duration > 18:
        penalties.append(f"时长稍欠({total_duration:.1f}s): -5")
        score -= 5

    # 2. 平均语速 (字/秒, 假设理想1.5-2.5字/秒)
    speaking_rate = row_features.get('speaking_rate_cps', 0)
    if speaking_rate == 0 : # 可能是无法检测到语音
        penalties.append(f"无法检测有效语音: -30")
        score -= 30
    elif speaking_rate < 1.2 or speaking_rate > 3.0:
        penalties.append(f"语速不佳({speaking_rate:.1f}cps): -15")
        score -= 15
    elif speaking_rate < 1.5 or speaking_rate > 2.5:
        penalties.append(f"语速稍欠({speaking_rate:.1f}cps): -5")
        score -= 5
        
    # 3. 流畅度 - 停顿次数 (语音段之间的停顿，理想情况是句末停顿，即 poem_num_lines - 1 = 3 次)
    # 这个简化的 num_pauses 可能不完全等于句末停顿，需要更复杂的逻辑
    num_pauses = row_features.get('num_pauses', 0)
    # 假设理想停顿次数是3 (4句诗，3个句间停顿)。允许2-4个主要停顿。
    if num_pauses < 2 or num_pauses > 4: # 允许更多停顿，但如果过多说明不流畅
        penalties.append(f"停顿次数不佳({num_pauses}): -10")
        score -= 10
    
    # 4. 平均音量 (RMS) - 需要根据实际数据分布设定理想范围
    avg_rms = row_features.get('avg_rms_energy', 0)
    # 假设我们通过 df_features.describe() 观察到大部分好的音频 avg_rms 在 0.05 - 0.2 之间
    if avg_rms < 0.02 or avg_rms > 0.3: # 声音太小或太大/爆麦 (需要调试)
        penalties.append(f"音量可能不佳({avg_rms:.2f}): -10")
        score -= 10

    # 5. 语音占比 (过低可能说明犹豫过多或静音过长)
    speech_ratio = row_features.get('speech_ratio', 0)
    if speech_ratio < 0.5 and speech_ratio > 0 : # 语音占比低于50%
        penalties.append(f"语音占比低({speech_ratio:.1f}): -10")
        score -= 10
        
    final_score = max(0, score) # 最低0分
    # print(f"File: {row_features['filename']}, Score: {final_score}, Penalties: {', '.join(penalties) if penalties else 'None'}")
    return final_score

# --- 应用打分 ---
# df_features['score'] = df_features.apply(score_reading, axis=1)
# print(df_features[['filename', 'score']].sort_values(by='score', ascending=False))

# --- 进一步思考与改进 ---
# 1. 参数调整：静音检测的 min_silence_len, silence_thresh 对特征提取影响巨大。
# 2. 句逗停顿的精确识别：目前的 num_pauses 比较粗略。可以尝试结合每句的预期时长来判断停顿是否合理。
#    例如，总时长D，每句平均 D/4。在 D/4, 2D/4, 3D/4 附近是否有合适的停顿？
# 3. 特征归一化与权重：在综合多个特征进行打分时，可能需要对特征进行归一化，并为不同特征设定权重。
# 4. 引入更多特征：如音高变化（反映是否有感情），特定字的清晰度（需要语音识别辅助，较难）。
# 5. 建立人工打分的小样本集：用人工打分的结果来验证和校准机器打分规则。
```

**重要提示给学习者：**
* **参数敏感性：** `pydub`中静音检测的`min_silence_len`和`silence_thresh`参数，以及`librosa`中某些特征提取的参数，需要根据实际音频的特点（如背景噪音水平、录音设备差异）进行仔细调试。没有通用的“最佳参数”。
* **打分规则的主观性与迭代：** 上述打分逻辑非常初步，仅为示例。实际的打分规则需要结合对“好”的朗读的教育学理解，并最好能有一小部分人工打分的样本作为参照，通过不断尝试和对比来优化。
* **错误处理：** 真实的音频文件可能存在各种问题（格式不支持、文件损坏、内容为空等），代码中需要加入更完善的错误处理机制。
* **从简单开始：** 先确保能稳定提取少数几个核心特征，并基于它们构建最简单的打分规则，然后再逐步增加特征和规则的复杂度。
