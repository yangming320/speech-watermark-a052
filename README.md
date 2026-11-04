# A052 语音扩频水印复现与跨码率检测

把已知扩频水印嵌入语音，经过降采样、噪声或 Opus 编码，再只凭接收音频、密钥和检测器计算水印分数。支持最长 30 秒 WAV。程序提供传统扩频、LPC 加权和 14 个预训练深度检测模型。

**[打开华为云演示](http://60.204.228.191:7860/)** · [下载 14 个演示权重及公开 WAV 样例](https://github.com/yangming320/speech-watermark-a052/releases/download/course-v1/a052-weights.zip) · [下载 M4～M6 实验权重](https://github.com/yangming320/speech-watermark-a052/releases/download/course-v1/a052-experiment-weights.zip)

## 助教复现步骤

下列命令在终端运行，适用于 macOS 和 Ubuntu 的 Bash/Zsh。演示使用 Python 3.12，无需 GPU、账号、训练语料或系统级 ffmpeg。Ubuntu 24.04 x86-64 需先安装 `python3-venv python3-dev build-essential libsndfile1` 和 CA 证书。macOS Apple 芯片使用原生 Python 3.12，若 PESQ 需要本机编译，请先准备 C 编译器。

### 1　获取代码与权重

```sh
git clone https://github.com/yangming320/speech-watermark-a052.git
cd speech-watermark-a052
curl -fL --retry 3 -o a052-weights.zip https://github.com/yangming320/speech-watermark-a052/releases/download/course-v1/a052-weights.zip
python3.12 -c "import hashlib,pathlib; p=pathlib.Path('a052-weights.zip'); h=hashlib.sha256(p.read_bytes()).hexdigest(); assert h == 'a6a754082c7270ea37b22a5d0476a80d00a2020031c892be4e882f4983dc8c3c', h; print('SHA-256 OK')"
python3.12 -m zipfile -e a052-weights.zip .
```

解压后应可看到 `deployment/weights/` 和 `deployment/samples/clean.wav`。展示包的密钥是可公开的固定实验序列。启动时还会逐项校验 14 个权重。可选实验包保留 M4、M5 和 M6 阶段目录及多个训练种子，校验值见 [SHA256SUMS](SHA256SUMS)。

### 2　安装

```sh
python3.12 -m venv .venv
. .venv/bin/activate
if [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ]; then
  python -m pip install -c requirements-lock.txt "$(python -c 'import json; print(json.load(open("deployment/linux-wheel-metadata.json"))["results"][1]["linux_cp312_x86_64_matches"][0])')"
fi
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps .
python -m pip check
```

### 3　核验模型并打开网页

```sh
python scripts/preflight_demo.py --root . --output preflight.json
python -m speechwm.demo --root . --manifest deployment/models.json --host 127.0.0.1 --port 7860
```

预检输出应为 `passed: true`、14 个模型和 8 种通道，JSON 同时保存 28 个固定输入对照。若再次预检，请换一个新的输出文件名。打开 [本机网页](http://127.0.0.1:7860/)，上传 `deployment/samples/clean.wav`，选择传统 SS、3 dB、Opus 8 kb/s、主实验模型 A、30 帧，点击「运行」。三秒样例应产生 9 组分数，首组约 **0.788454**。下载「实际检测输入」的 WAV，新开页面选择「独立检测上传音频」，使用相同模型和帧数，九组结果应一致。试听文件使用 PCM16，精确重放应选择实际检测输入的浮点 WAV。

程序在任意上传时返回分数；概率、通用二分类阈值及说话人身份均不由这些分数给出。服务采用单任务模式。

### 4　自动回归与复核

```sh
python -m pytest -q
```

如需自动检查真实 HTTP 往返，可以在空闲端口启动服务时增加 `--ready-file local-ready.json`，再在同一虚拟环境的第二个终端运行 `python scripts/verify_demo_http.py --root . --ready-file local-ready.json --output http-check`。输出目录应为新目录，ready 文件仅留在本机。

## 实验与证据

用有许可的 LibriSpeech 替代语料进行方法复现，训练、验证、测试分别有 288/80/200 句和 24/10/40 位说话人。M5 主实验与 M6 探索使用同一固定测试子集。Opus 8 kb/s、30 个有效帧下，传统相关、单码率模型 A 与混合码率模型 A 的等错误率分别为 20.909%、15.510% 和 14.562%。6 kb/s 的相应结果为 26.205%、21.780% 和 21.358%。

- [主实验汇总](results/m5-seed-summary.csv) · [跨码率与扩展汇总](results/m6-seed-summary.csv) · [评估协议](docs/)
- [云端预检](milestones/M7/linux-preflight.json) · [公网端到端结果](milestones/M7/public-verification.json) · [云主机环境](milestones/M7/runtime.txt)
- [Word 报告](deliverables/A052-Report.docx) · [可编辑 PPT](deliverables/A052-Presentation.pptx) · [九分钟讲稿](deliverables/A052-Talk.md)

如需重建正式数据，可在仓库根目录运行 `python scripts/fetch_formal_data.py` 和 `python -m speechwm.cli prepare-formal`。公开数据按官方哈希校验，数据来源与署名见 [语料说明](data/SOURCES.md)，公开示例音频的 CC BY 4.0 许可随演示包提供。复现论文为 Looney 与 Gaubitch 的 *Robust Spread Spectrum Speech Watermarking Using Linear Prediction and Deep Spectral Shaping*，见 [Interspeech 2024 论文页](https://www.isca-archive.org/interspeech_2024/looney24_interspeech.html)。
