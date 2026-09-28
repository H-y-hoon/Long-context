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

아래 1–7단계를 스크립트로 자동화했습니다. 서버마다 `setup.sh`를 한 번 실행한 뒤 `run_eval.sh`로 평가합니다.

```bash
scripts/setup.sh                                 # NIAH clone(고정 commit) + 파일 복사/패치 + venv + 의존성
scripts/run_eval.sh --model google/gemma-4-E4B-it --context_length 4 8 16
scripts/run_eval.sh --adapter /path/to/lora --context_length all --tokenizer model --gpus 0,1
scripts/run_eval.sh --help                       # 전체 옵션
```

- `setup.sh`는 드라이버의 CUDA 버전을 보고 PyTorch wheel(cu126/cu128)을 고릅니다. `--torch_index`로 직접 지정할 수 있고, GPU가 없으면 cpu wheel을 설치합니다. 다른 서버에서 복사해 온 `.venv`는 해당 서버의 경로를 가리키므로 새로 만듭니다. 다시 실행해도 안전합니다.
- `--adapter`만 주면 base 모델은 어댑터의 `adapter_config.json`에 있는 `base_model_name_or_path`에서 가져옵니다. 어댑터를 주지 않으면 사전학습 모델 그대로 평가합니다.
- `--tokenizer`는 context 길이를 세는 기준입니다.
  - `gpt`(기본값): NIAH 원래 기준인 `cl100k_base`로 셉니다. 같은 길이라도 모델 토큰으로는 더 길어질 수 있으므로, 모델의 최대 길이를 넘는 길이는 건너뜁니다(Gemma 4 E4B는 128k).
  - `model`: 평가하는 모델의 토크나이저로 셉니다. 128k는 `입력 + max_new_tokens`가 모델의 최대 길이에 들어가도록 조금 줄여서 실행합니다.
- 결과는 `needle-in-a-haystack/results/<모델>[__<어댑터>]__tok-<토크나이저>/`에 저장됩니다. 이 폴더에는 `results.jsonl`, `summary.csv`, 생성된 `run.yaml`/`model.yaml`, `plan.json`이 들어갑니다. 같은 명령을 다시 실행하면 이미 끝난 조합은 건너뜁니다.
- `--tokenizer model`로 실행한 결과를 `niah reconstruct`로 복원하려면, `plan.json`의 `length_tokenizer` 값을 `NIAH_TOKENIZER` 환경변수로 설정한 뒤 실행해야 합니다.

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
