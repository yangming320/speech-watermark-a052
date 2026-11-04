# 语料来源与使用说明

用户确认暂时没有 TIMIT，同意先用公开小样本。下述三个 HQ Ogg 示例用于 M1～M4 smoke 开发；M5 新增的独立替代语料见后文，旧 smoke 清单保持原样。

## 来源与署名

音频来自 librosa 官方示例的三个 LibriSpeech HQ Ogg 文件。音频及所附署名文本均按官方注册表 SHA-256 校验。

| 文件标识 | 说话人标识 | 朗读者 | 原片段时长 |
| --- | --- | --- | ---: |
| 5703-47212-0000 | 5703 | Garth Comira | 14.840 秒 |
| 3436-172162-0000 | 3436 | Anders Lankford | 16.745 秒 |
| 198-209-0000 | 198 | Heather Barnett | 13.910 秒 |

三个文件所附文本均标注来自 [LibriSpeech SLR12](https://www.openslr.org/12/)，采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。原始署名文本保存在 `raw/public/` 的相应 `.txt` 文件中。

数据说明见 [librosa 官方示例](https://librosa.org/doc/main/recordings.html)，哈希见 [官方注册表](https://raw.githubusercontent.com/librosa/librosa/main/librosa/util/example_data/registry.txt)。固定下载地址及完整校验值保存在 `configs/public_sources.json`。

## 本项目的处理

源文件实际均为 16 kHz 单声道 Ogg，是已有有损编码的公开示例，并非原始无损 LibriSpeech 或 TIMIT。

读取后采用只降低的输入增益，将每条源语音分别等分为 7、7、6 个不重叠片段，保存为 16 kHz 浮点 WAV。20 个片段总计 45.4950625 秒。实验进一步生成嵌入、重建与降采样版本，均属于本项目的修改。

每个片段保留源语音、说话人、原时间范围、URL、哈希和处理增益。清单见 `manifests/smoke.jsonl`，汇总见 `manifests/data_summary.json`。

20 个片段共享三条源语音，不能当成 20 条独立采集语音或随机分散到训练集、测试集。后续取得正式语料后另建说话人划分，不把本次 smoke 结果改名为正式测试结果。分发音频或衍生样本时保留署名、许可证链接与修改说明。


## M5：LibriSpeech 固定替代语料子集

用户要求继续开展主实验，沿用暂无 TIMIT 的条件，按实施方案采用公开替代语料。原始来源为 [OpenSLR SLR31](https://www.openslr.org/31/) 的 train-clean-5/dev-clean-2 与 [SLR12](https://www.openslr.org/12/) 的 test-clean；全部遵循 CC BY 4.0。原始朗读者署名、官方性别标签和 corpus 分区在各池的 SPEAKERS.TXT，选中朗读者姓名也随 all.jsonl 保存；不猜测性别。

三个包已核对官方 MD5，实际 SHA-256 为：

| 包 | SHA-256 |
| --- | --- |
| train-clean-5.tar.gz | 47805806c8b15f7549f3c51bd6b7da72ec0128a34d32830c8014a88658c0292d |
| dev-clean-2.tar.gz | 176ec501490eced2d6c1f89f4f0ddc7dfe799e649e5322f8ba49fe3ff50c8012 |
| test-clean.tar.gz | 39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23 |

源目录索引实际为 1519/1089/2620 条语句。按种子 20261018 的固定哈希排序、官方性别元数据及长度至少 3 秒筛选，训练取 24 位说话人各 12 句（288），验证取另 10 人各 8 句（80），测试取另 40 人各 5 句（200）。每条源语句只取一个居中 3 秒片段。总计 568 条、74 位不同说话人、1704 秒；测试 F/M 各 100 条，分别来自 20 位说话人。

排除旧 smoke 说话人 5703、3436、198，按说话人/父语句/源文件/原始 PCM/裁剪 PCM 检查隔离。选样不依赖 VAD、PESQ 或检测分数。清单与冻结信息位于 `manifests/librispeech_m5_v1/`；策略在 [formal_data.yaml](../configs/formal_data.yaml)，详细协议在 [M5评估协议](../docs/M5评估协议.md)。

选中的原始 FLAC 与共享官方元数据放 `raw/librispeech_m5_v1/`，准备片段放 `processed/librispeech_m5_v1/`。dev-clean-2 包本身不附全局元数据，使用已校验训练包中的共享 LICENSE/SPEAKERS/CHAPTERS，并在 METADATA_SOURCE.json 逐项注明来源和 SHA；未伪造缺失的验证 README。三个全量下载 tar 留在本地缓存，归档包含所有实际选用的原始文件/许可证/元数据和准备片段。

分发格式是无损 FLAC，LibriSpeech 来自 LibriVox 朗读材料，录音条件、训练规模和居中裁剪与 TIMIT 不同。本轮数据用于项目的独立替代语料主实验，不能描述成原论文的同条件 TIMIT 或完整 LibriSpeech benchmark。所有浮点增益、水印、噪声和 Opus 音频都是本项目的衍生修改，保留来源及修改说明。
