# Transformer-based Fine-Grained Emotion Recognition
This repository contains code for applying advanced language transformer fine-tuning methods to recognize emotions using the GoEmotions dataset.
## Implemented Functionalities
- Layer-wise learning rate decay
- Reinitialization of the last N layers
- Linear learning rate decay scheduler with initial warmup
- Last Epoch Frequent Evaluation
- Special tokens '[RELIGION]' and '[NAME]' masks
- Storage of experiment metrics and parameters
- Preprocessed version of GoEmotions dataset without accents, special characters, emojis, and emoticons
## Usage
Before you begin, ensure that you have Python 3.8 and VirtualEnv installed in your working environment.
1. Run `make install` to create a virtual environment with the project dependencies.
2. Activate the virtual environment using `source .venv/bin/activate`.
3. Execute the `main.py` script with the desired parameters and parameter group ID.
4. Once the optimization loop is complete, you can find all the experiment metrics and parameters for this run based on the `--params-id` and desired `--output-dir` configuration.
### Training Args
The following table lists the command-line arguments that can be used with `main.py`:
| Argument                 | Type       | Default            | Description                                                                                               |
|--------------------------|------------|--------------------|-----------------------------------------------------------------------------------------------------------|
| --dev-run                | boolean    | False              | Flag to run in dev mode                                                                                    |
| --params-id              | str        | None (required)    | Identifier for the hyperparameter config being experimented                                               |
| --data-dir               | str        | "data"             | Path to the data directory                                                                                 |
| --output-dir             | str        | "experiments"      | Path to the data directory                                                                                 |
| --dataset                | str        | "raw"              | Either "raw" or "preprocessed"                                                                              |
| --model-name             | str        | "distilbert-base-uncased" | Huggingface pretrained model name                                                                  |
| --batch-size             | int        | 32                 | Batch size                                                                                                |
| --epochs                 | int        | 3                  | Number of epochs                                                                                           |
| --llrd-init-lr           | float      | 5e-5               | Initial learning rate for the last layer and head in the LLRD                                             |
| --reinit-n-layers        | int        | 2                  | Number of last transformer layers to reinitialize (ignored if do-not-reinit-layers is TRUE)                |
| --do-not-reinit-layers   | boolean    | False              | Flag to deactivate layers reinitialization                                                                 |
| --llrd-mult-factor       | float      | 0.9                | LLRD multiplication factor                                                                                 |
| --weight-decay           | float      | 1e-2               | Weight decay                                                                                              |
| --warmup-steps-ratio     | float      | 0.1                | Ratio of total steps to perform warmup                                                                     |
| --freeze-pretrained      | boolean    | False              | Flag to freeze pretrained layers                                                                           |
| --additional-special-tokens | list[str] | ["[NAME]", "[RELIGION]"] | Additional special tokens                                                                             |
| --padding                | str        | "max_length"       | Tokenizer padding strategy                                                                                 |
| --truncation             | boolean    | True               | Flag to force tokenizer to truncate                                                                        |
| --return-tensors         | str        | "pt"               | Tokenizer type of tensors to return                                                                        |
| --max-length             | int        | 50                 | Tokenizer maximum sequence length in tokens                                                                |
| --freq-eval-iters        | int        | 10                 | Number of times to evaluate on the last epoch                                                             |
| --dropout                | float      | 0.35               | Classifier dropout                                                                                        |
| --seed                   | float      | 1                  | Seed used to make results reproducible                                                                     |

