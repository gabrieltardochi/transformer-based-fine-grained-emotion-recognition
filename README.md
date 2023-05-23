# transformer based fine grained emotion recognition
Code to apply advanced language transformers fine-tuning methods to recognize emotions using the GoEmotions dataset.
## Implemented Functionalities
- Layer-wise learning rate decay
- Last *N* layers reinitialization
- Linear learning rate decay scheduler with initial warmup
- Last Epoch Frequent Evaluation
- '[RELIGION]' and '[NAME]' masks as special tokens
- Storage of experiment metrics and params
- A preprocessed version of GoEmotions without accents, special characters, emojis and emoticons
## Development
First of all, make sure you have Python 3.8 anv VirtualEnv installed in your working environment.
1. Run `make install` to create a virtualenv with this project dependencies;
2. Activate the venv with `source .venv/bin/activate`;
3. Execute the python script `main.py` with the desired parameters and parameter group id;
4. Once the full optimization loop ends, you will be able to find, based on your experiment `--params-id` and desired `--output-dir`, every single metric and parameter configured for this run.
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

