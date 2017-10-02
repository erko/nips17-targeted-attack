# NIPS 2017: Targeted Adversarial Attack submission
This project is final submission on kaggle competition [NIPS 2017: Targeted Adversarial Attack](https://www.kaggle.com/c/nips-2017-targeted-adversarial-attack) by team ["erko" (Yerkebulan Berdibekov)](https://www.kaggle.com/erkowa)

*Description of idea here*

Requirements:

Configure on host OS CUDA & cuDNN 6, docker-ce, nvidia-docker.

### Instructions:
* Download dev_toolkit - https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack/data & extract to dev_toolkit folder;
* Download developement set - https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set and extract into dev_toolkit/dataset/;
* Inside of developement toolkit prepare sample_attacks, sample_targeted_attacks, sample_defenses by downloading checkpoints. Run `sh download_checkpoints.sh` in this folders;
* Put this project in *new folder* in sample_targeted_attacks;
* In this *new folder*, download checkpoint files: run **`sh download_checkpoints.sh`**;
* Edit file `run_attacks_and_defenses.sh`: append argument `--gpu` in line executing python script;
* Run `sh run_attacks_and_defenses.sh`;
* Compare this targeted-attack to other targeted-attacks in `accuracy_on_targeted_attacks.csv` file in resulting output folder.

Possible problems: 
- Facing errors: "****.sh: Permission denied!" - may be need to make .sh file runnable.
