# Individual Project Proposal — Kyrgyz Voice Command Recognition on a Small Device

**Student:** Ersultan Ismailov **Student ID:** 70662 **Course:** Machine Learning and Its Applications (Group 01)

## 1. Problem

I want to build a small model that hears a short spoken word in **Kyrgyz** and recognises it.
The model listens to a 1-second audio clip and says which of about 10 chosen Kyrgyz words it was.
It can also say "silence" (nobody is talking) or "unknown" (some other word). The model must work **offline on a normal CPU** (a laptop or a small board like a Raspberry Pi), without a GPU
and without the internet. Voice assistants almost never support Kyrgyz, so a small offline model for my language is a useful and realistic task.

## 2. Dataset

**Multilingual Spoken Words Corpus (MSWC), Kyrgyz part** (MLCommons, CC-BY 4.0).
MSWC is a public dataset made for keyword spotting. It has 1-second clips of single spoken words in
50 languages. The words were cut automatically from Mozilla Common Voice sentences.
The Kyrgyz part has about **12 hours** of audio (about 0.1 GB). Load: `load_dataset("MLCommons/ml_spoken_words", "ky_wav")`.

- **Classes:** I will choose about 10 words that have the most recordings, and if possible words that work as commands (for example "ооба" – yes, "жок" – no, numbers "бир", "эки", "үч"). Final list after counting the data. Plus "unknown" (other Kyrgyz words) and "silence" (background noise clips) → about 12 classes.
- **Split:** I use the official train / dev / test split and check that one speaker is not in two parts (no data leakage, like in Lab 4).
- **Pre-processing:** resample to 16 kHz, turn each clip into a spectrogram (a picture of the sound), add background noise and small time shifts so the model works in real conditions.
- **Backup:** if some words have too few examples, I will cut more examples from Common Voice Kyrgyz (~48 h).

## 3. Model

- **Baseline:** a small CNN, similar to my Lab 3 model.
- **Main model:** a DS-CNN, a very light CNN made for small devices (25–100 thousand parameters).
- **Teacher model:** a bigger CNN that is only used to help train the small one (see section 6).
  All models are trained with my Lab 6 pipeline (YAML config, MLflow, checkpoints, profiling).

## 4. How I measure success

| What I measure                                        | Goal                                                               |
| ----------------------------------------------------- | ------------------------------------------------------------------ |
| Accuracy on the test set                              | at least 85 %, and less than 1.5 % lost after optimisation         |
| Time to recognise one clip on CPU (incl. spectrogram) | **50 ms or less**                                                  |
| Model file size                                       | **5 MB or less** (I hope for less than 0.5 MB)                     |
| Extra                                                 | confusion matrix (which words get mixed up), energy per prediction |

The accuracy goal is lower than for English datasets because the Kyrgyz data is smaller and its labels
were made automatically, so some clips may be wrong.

## 5. Deployment constraints

The model runs on a CPU, one clip at a time, with ONNX Runtime. Hard limits: **50 ms per prediction**
and **5 MB model size**. To check the speed, I run the model 1,000 times and report the typical and the slowest results. For the demo I will make a small web API (FastAPI).

## 6. Planned optimisation

1. **Quantization (main step):** store the weights as 8-bit numbers instead of 32-bit. The model becomes about 4 times smaller and usually faster. If accuracy drops too much, I will use quantization-aware training.
2. **Distillation (extra step):** the big teacher model "teaches" the small model, so the small model gets back some of the accuracy it lost by being small. After each step I compare accuracy, speed and size with the original model.

## 7. Plan and risks

**Plan:** Week 1 — download data, choose words, baseline. Week 2 — DS-CNN and teacher. Week 3 — quantization. Week 4 — distillation, speed tests, demo. Week 5 — final report.
**Risks:**

- Few examples for some words → choose the most frequent words, use augmentation, add clips from Common Voice.
- Wrong labels from automatic cutting → listen to a random sample of clips and remove bad ones.
- Many more "unknown" clips than other classes → balance the classes during training.
- Accuracy drop after quantization → quantization-aware training.
