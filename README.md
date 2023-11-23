# TTS project

## Vocoder

To download vocoder, add
```json
"vocoder": {
        "path": "<path_to_be_placed>"
},
```
to config.json
It will be waveglow v2 trained on ljspeech.

## Graphemes and alignments

I used pretrained MFA. You might download it [here]()
or just add
```json
"preprocessing": {
   ...
   "g2p": true,
   ...
}
```
to config. It will be placed in the same directory, as your provided dataset.

## Audio validation

- `A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`
- `Massachusetts Institute of Technology may be best known for its math, science and engineering education`
- `Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`

These examples of synthezed audio are provided in directory `test_data`

## Report

You may see the report if you follow [this link](https://wandb.ai/l0u1za/tts_project/reports/TTS-1--Vmlldzo2MDU2MDgx?accessToken=jgjjaaem4mrdmkg1zq0h6e1f7mfcsja31hsiti2h7ox1gva4tdexincbe45ugtrr)

## Installation guide

Firstly, install needed requirements for running model

```shell
pip install -r ./requirements.txt
```

### Download model

Use bash script to download trained model

```shell
cd ./default_test_model
./download.sh
```

It will be placed to `./default_test_model/checkpoint.pth`

If you have some issues using bash utilities, you may download model directly from [google drive](https://drive.google.com/file/d/1Q4diwFZktzevEBnocQcIPRm03JPnu8GJ/view?usp=sharing)

## Run test model with prepared configuration

```shell
python test.py \
   -c default_test_config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash
docker build -t my_src_image .
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_src_image python -m unittest
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize
