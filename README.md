# Long Context Gemma4 Evaluation

이 디렉터리는 공개 NIAH(`gkamradt/needle-in-a-haystack`) 저장소에 Gemma4 실행 파일을 추가해 `google/gemma-4-E4B-it` long-context 평가를 실행하는 방법을 안내합니다. NIAH 원본 전체를 TANGO2에 복사하지 않고, 필요한 추가 파일만 제공합니다.

## 포함 파일

```text
Long_Context/
├── README.md
├── files/
│   ├── configs/models/google-gemma-4-e4b-it-transformers.yaml
│   ├── configs/runs/gemma_single_needle.local.yaml
│   └── needlehaystack/providers/transformers_local.py
├── patches/
│   ├── register_transformers_provider.patch
│   └── configurable_tokenizer.patch
└── scripts/
    ├── setup.sh
    ├── run_eval.sh
    └── niah_helper.py
```

## 빠른 사용 (스크립트)

아래 1–7단계의 수동 절차를 스크립트로 자동화했습니다. 서버마다 `setup.sh`를 한 번 실행한 뒤, `run_eval.sh` 맨 위의 설정만 고쳐서 실행합니다.

### 1) 설치

```bash
git clone https://github.com/H-y-hoon/Long-context.git
cd Long-context
scripts/setup.sh
```

`setup.sh`가 하는 일은 다음과 같습니다. 다시 실행해도 안전합니다.

- NIAH 원본을 `needle-in-a-haystack/`에 고정 commit으로 clone하고, `files/`를 복사하고 `patches/`를 적용합니다.
- uv가 관리하는 Python 3.12로 `.venv`를 만듭니다. 다른 서버에서 복사해 온 `.venv`는 그 서버의 경로를 가리키므로 새로 만듭니다.
- 드라이버의 CUDA 버전을 보고 PyTorch wheel(cu126/cu128)을 고릅니다. `--torch_index`로 직접 지정할 수 있고, GPU가 없으면 cpu wheel을 설치합니다.
- 테스트한 버전으로 고정된 의존성을 설치하고, `cl100k_base` 토크나이저를 미리 받아 둡니다. 그래서 인터넷이 막힌 서버에서도 동작합니다.

Gemma 모델을 받으려면 Hugging Face 로그인과 모델 이용 동의가 필요할 수 있습니다.

```bash
source needle-in-a-haystack/.venv/bin/activate
hf auth login
```

### 2) 어댑터 다운로드 (선택)

학습한 LoRA 어댑터는 Hugging Face Hub에 public으로 올려 두었습니다. 사전학습 모델만 평가한다면 이 단계는 건너뜁니다.

| HF 레포 | base 모델 | 크기 |
|---|---|---|
| [yanghoon/gemma4-e4b-ipo-v1-lora](https://huggingface.co/yanghoon/gemma4-e4b-ipo-v1-lora) | `google/gemma-4-E4B-it` | 약 300MB |
| [yanghoon/gemma4-12b-ipo-v1-lora](https://huggingface.co/yanghoon/gemma4-12b-ipo-v1-lora) | `google/gemma-4-12B-it` | 약 550MB |

`Long-context` 폴더에서 실행합니다. `hf` 명령은 `.venv` 안에 설치되어 있습니다.

```bash
source needle-in-a-haystack/.venv/bin/activate
hf download yanghoon/gemma4-e4b-ipo-v1-lora --local-dir ./adapters/gemma4-e4b-ipo-v1-lora
hf download yanghoon/gemma4-12b-ipo-v1-lora --local-dir ./adapters/gemma4-12b-ipo-v1-lora
```

- `--local-dir`를 빼면 HF 캐시(`~/.cache/huggingface/hub/...`) 안의 긴 경로에 저장됩니다. 찾기 쉬운 위치로 지정하는 것을 권합니다.
- `--local-dir`는 명령을 실행한 폴더 기준입니다. 다른 폴더에서 실행한다면 `/`로 시작하는 절대경로로 적습니다.
- `adapters/`는 `.gitignore`에 들어 있어서 git에 올라가지 않습니다.
- 제대로 받았는지 확인하려면 폴더마다 base 모델이 맞게 나오는지 봅니다.
  ```bash
  grep base_model_name_or_path adapters/*/adapter_config.json
  ```

### 3) 평가 설정

`scripts/run_eval.sh` 맨 위의 설정 블록만 고칩니다.

```bash
MODEL="google/gemma-4-E4B-it"   # HF 모델 id 또는 로컬 경로. ADAPTER를 쓸 때 ""이면 adapter_config.json에서 읽음
ADAPTER=""                      # 어댑터 폴더(절대경로). "" = 사전학습 모델 그대로 평가
CONTEXT=(all)                   # k 단위. 예: (2 4) / (4 8 16). (all) = 4 8 16 32 64 128
TOKENIZER="gpt"                 # 길이를 세는 기준: gpt | model
DEPTHS=(50)                     # needle 위치(%). 예: (0 50 100)
GPUS=""                         # 예: "0" / "0,1". "" = 보이는 GPU 전부
```

예를 들어 E4B 어댑터를 2k·4k에서 평가하려면 다음과 같이 적습니다.

```bash
MODEL=""
ADAPTER="/home/<user>/.../Long-context/adapters/gemma4-e4b-ipo-v1-lora"
CONTEXT=(2 4)
TOKENIZER="model"
DEPTHS=(0 50 100)
GPUS="0,1"
```

- `CONTEXT=(...)`, `DEPTHS=(...)`는 bash 배열이므로 괄호를 유지합니다. `=` 앞뒤에는 공백을 넣지 않습니다.
- `ADAPTER`는 절대경로로 적습니다. 상대경로는 명령을 실행한 폴더를 기준으로 해석됩니다.
- `TOKENIZER`는 context 길이를 세는 기준입니다. 답변 생성에는 항상 모델의 토크나이저를 씁니다.
  - `gpt`(기본값): NIAH 원래 기준인 `cl100k_base`로 셉니다. 같은 길이라도 Gemma 토큰으로는 약 3.6% 더 길어서, 모델의 최대 길이를 넘는 길이(Gemma 4 E4B의 128k)는 건너뜁니다.
  - `model`: 평가하는 모델의 토크나이저로 셉니다. 어댑터 폴더에 토크나이저가 있으면 어댑터의 토크나이저를 씁니다. 128k는 `입력 + MAX_NEW_TOKENS`가 모델의 최대 길이에 들어가도록 조금 줄여서 실행합니다.
- 명령줄 옵션을 주면 그 실행에서만 설정 블록 값을 덮어씁니다. 예: `scripts/run_eval.sh --context_length 4 --dry_run`. 전체 옵션은 `--help`로 볼 수 있습니다.

### 4) 실행

```bash
bash scripts/run_eval.sh --dry_run   # 모델을 불러오지 않고 길이 계산과 설정만 확인
bash scripts/run_eval.sh
```

dry run은 길이별로 모델이 실제로 받을 토큰 수를 출력합니다. 예: `4k: context_length=4,096 -> prompt 4,139 model tokens`. 실제 실행에서는 한 번 평가할 때마다 `[ok] ctx=4096 depth=50.0% score=1.00`처럼 한 줄씩 출력하고, 마지막에 길이 × 위치 요약 표를 보여 줍니다.

### 5) 결과

결과는 `needle-in-a-haystack/results/<모델>[__<어댑터>]__tok-<토크나이저>/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `results.jsonl` | 평가한 조합별 원본 결과(응답, 점수, 토큰 수) |
| `summary.csv` | 길이 × 위치 점수 표 |
| `run.yaml`, `model.yaml` | 스크립트가 만든 NIAH 설정 |
| `plan.json` | 길이 라벨과 길이를 센 토크나이저 기록 |

- 점수는 응답에 `eat a sandwich and sit in Dolores Park`가 포함되면 1, 아니면 0입니다(대소문자 무시). 표현이 조금만 달라도 0점이 되므로, 0점이 나오면 `results.jsonl`의 `response`를 확인합니다.
- 같은 설정으로 다시 실행하면 이미 끝난 조합은 건너뜁니다. 오류가 난 조합도 끝난 것으로 치므로, 다시 평가하려면 해당 결과 폴더를 지웁니다.
- `TOKENIZER="model"`로 실행한 결과를 `niah reconstruct`로 복원하려면, `plan.json`의 `length_tokenizer` 값을 `NIAH_TOKENIZER` 환경변수로 설정한 뒤 실행해야 합니다.

### 6) GPU 메모리 주의

`run_eval.sh`는 모델을 `device_map="auto"`로 불러옵니다. 모델을 불러오는 순간 각 GPU의 남은 메모리를 보고 **가중치만** 여러 GPU에 나눠 올립니다. 긴 context에서 커지는 attention 계산 메모리는 나눠지지 않고, 해당 layer가 올라간 GPU 한 장에서 전부 씁니다. Gemma 4 12B는 전체 attention layer의 head 크기가 512입니다. PyTorch SDPA의 메모리 효율적인 계산 방식이 이 크기를 지원하지 않아서, 입력 길이의 제곱에 비례하는 메모리를 씁니다.

`gemma-4-12B-it`를 H200(140GB) 한 장에서 측정한 값입니다. 가중치는 22.3GiB입니다.

| 입력 길이 | 최대 사용량 | 가중치를 뺀 추론 메모리 |
|---|---|---|
| 16k | 30.0 GiB | +7.7 GiB |
| 32k | 39.6 GiB | +17.3 GiB |
| 64k | 64.9 GiB | +42.6 GiB |
| 128k | OOM | attention 한 번에 31.9 GiB 요청 |

- 24–32GB GPU 여러 장에서는 12B의 32k 이상이 OOM이 날 가능성이 높습니다. 긴 context는 메모리가 큰 GPU에서 평가합니다.
- 다른 작업과 GPU를 같이 쓰는 경우, `GPUS`에는 남은 메모리가 많은 GPU만 지정합니다. 모델을 불러온 뒤에 상대 작업이 메모리를 더 쓰면 평가 도중에 OOM이 날 수 있습니다.
- OOM이 난 조합은 `error`로 기록되고, 평가는 다음 조합으로 넘어갑니다.

## 1. NIAH 원본 저장소 준비

실험을 실행할 위치에서 NIAH 원본 저장소를 clone하고 기본 개발 환경을 만듭니다.

```bash
git clone https://github.com/gkamradt/needle-in-a-haystack.git
cd needle-in-a-haystack
uv sync --extra dev
```

기본 CLI가 정상인지 먼저 확인합니다.

```bash
uv run pytest
uv run niah run configs/runs/smoke.fake.yaml
```

## 2. Gemma4 추가 파일 복사

아래 경로는 TANGO2 clone 안의 `Long_Context` 디렉터리를 가리키도록 바꿉니다.

```bash
TANGO2_LC=/path/to/TANGO2/LLMOps/Evaluation/Long_Context
```

Gemma4 model config, run config, local Transformers provider를 NIAH clone의 동일 경로로 복사합니다.

```bash
cp "$TANGO2_LC/files/configs/models/google-gemma-4-e4b-it-transformers.yaml" \
  configs/models/google-gemma-4-e4b-it-transformers.yaml

cp "$TANGO2_LC/files/configs/runs/gemma_single_needle.local.yaml" \
  configs/runs/gemma_single_needle.local.yaml

cp "$TANGO2_LC/files/needlehaystack/providers/transformers_local.py" \
  needlehaystack/providers/transformers_local.py
```

NIAH provider registry에 local Transformers provider를 등록합니다.

```bash
git apply "$TANGO2_LC/patches/register_transformers_provider.patch"
git apply "$TANGO2_LC/patches/configurable_tokenizer.patch"   # NIAH_TOKENIZER 지원(미설정 시 기존 동작)
```

## 3. Gemma4 의존성 설치

아래 명령은 1단계에서 만든 NIAH `uv` 환경 위에 Gemma4 실행 의존성을 추가로 설치합니다. CUDA 버전이 다르면 PyTorch 공식 안내에 맞춰 `--index-url`을 바꿉니다.

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
uv pip install transformers accelerate sentencepiece protobuf pillow tokenizers==0.22.2
```

`uv pip install`로 런타임 의존성을 추가한 뒤에는 `uv run --no-sync`를 사용합니다. 일반 `uv run`은 환경을 다시 동기화하면서 `tokenizers` 버전을 되돌릴 수 있습니다.

Hugging Face 접근 권한이 필요한 환경이면 먼저 로그인합니다.

```bash
hf auth login
```

## 4. 설정 확인

Gemma4 모델 설정은 다음 파일에 있습니다.

```text
configs/models/google-gemma-4-e4b-it-transformers.yaml
```

평가 run 설정은 다음 파일에 있습니다.

```text
configs/runs/gemma_single_needle.local.yaml
```

기본 평가 범위는 `context_lengths=[4096, 8192, 16384, 32768]`, `depth_percents=[50]`, `seeds=[1]`입니다. GPU 메모리가 부족하면 `context_lengths`를 줄여 먼저 확인합니다.

## 5. Dry-run

실제 모델 실행 전에 설정과 provider 등록이 정상인지 확인합니다.

```bash
uv run --no-sync niah validate configs/runs/gemma_single_needle.local.yaml
uv run --no-sync niah run configs/runs/gemma_single_needle.local.yaml --dry-run
```

## 6. 평가 실행

특정 GPU를 지정하려면 `CUDA_VISIBLE_DEVICES`를 사용합니다.

```bash
CUDA_VISIBLE_DEVICES=2 uv run --no-sync niah run configs/runs/gemma_single_needle.local.yaml
```

결과는 다음 파일에 저장됩니다.

```text
results/gemma-4-e4b-it-single-needle.jsonl
```

## 7. 결과 확인

JSONL row 수를 확인합니다.

```bash
wc -l results/gemma-4-e4b-it-single-needle.jsonl
```

특정 row에서 모델이 실제로 받은 context를 복원하려면 다음 명령을 사용합니다.

```bash
uv run --no-sync niah reconstruct results/gemma-4-e4b-it-single-needle.jsonl --row 0
```

## 8. 문제 해결

`Gemma4Processor requires the PIL library` 오류가 나면 `pillow`가 빠진 것입니다.

```bash
uv pip install pillow
```

`No module named 'torchvision'` 오류가 나면 `torchvision`이 빠진 것입니다.

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 torchvision
```

CUDA driver mismatch 오류가 나면 현재 드라이버와 맞는 PyTorch wheel을 다시 설치합니다. 기본 `torch` wheel이 현재 드라이버보다 높은 CUDA 버전으로 설치되면 실행이 실패할 수 있습니다.
