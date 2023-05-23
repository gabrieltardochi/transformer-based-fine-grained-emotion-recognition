import torch.nn as nn
from transformers import AutoModel


class EmotionRoBERTa(nn.Module):
    """Load roberta from hf and add a classification layer that maps"""

    def __init__(self, model, dropout, hidden_size, labels):
        super(EmotionRoBERTa, self).__init__()

        self.transformer_model = AutoModel.from_pretrained(model)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, labels),
        )  # traditionally implemented like this, using the pooler output with a linear classifier on top

    def forward(self, inputs):
        out = self.transformer_model(**inputs, return_dict=True)["pooler_output"]
        logits = self.classifier(out)
        return logits

    def _do_reinit(self, n_layers):
        # Re-init pooler.
        self.transformer_model.pooler.dense.weight.data.normal_(
            mean=0.0, std=self.transformer_model.config.initializer_range
        )
        self.transformer_model.pooler.dense.bias.data.zero_()
        for param in self.transformer_model.pooler.parameters():
            param.requires_grad = True
        # Re-init last n layers
        for n in range(n_layers):
            self.transformer_model.encoder.layer[-(n + 1)].apply(
                self._init_weight_and_bias
            )

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.transformer_model.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class EmotionDistilBERT(nn.Module):
    def __init__(self, model, dropout, hidden_size, labels):
        super(EmotionDistilBERT, self).__init__()

        self.transformer_model = AutoModel.from_pretrained(model)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, labels),
        )  # traditionally implemented like this, with a pre classifier and a classifier layer on top

    def forward(self, inputs):
        out = self.transformer_model(**inputs, return_dict=True)["last_hidden_state"][
            :, 0
        ]  # DistilBERT output for the [CLS] token
        logits = self.classifier(out)
        return logits

    def _do_reinit(self, n_layers):
        # Re-init last n layers
        for n in range(n_layers):
            self.transformer_model.transformer.layer[-(n + 1)].apply(
                self._init_weight_and_bias
            )

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.transformer_model.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
