
import torch
from torch.nn import BCEWithLogitsLoss

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    # BertConfig,
    get_linear_schedule_with_warmup,
)

from constants import MODEL_FOLDER

import logging

import random
import numpy as np

from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


class InputExample():
    def __init__(self, guid, text_a, text_b=None, labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeature():
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class BertClassifier():
    def __init__(
        self,
        no_cuda=False,
        bert_tokenizer_type='bert-base-uncased',
        seed=514,
        learning_rate=5e-5,
        warmup=0.1,
        batch_size=64,
        gradient_accumulation_steps=1,
        num_epochs=3,
        max_seq_length=128,
        do_lower_case=True,
        num_labels=2,
        debug_steps=0,
        do_pad=True,
        progress=False,
    ):
        self.output_mode = 'classification'
        self.no_cuda = no_cuda
        self.bert_tokenizer_type = bert_tokenizer_type
        self.seed = seed
        self.learning_rate = learning_rate
        self.warmup = warmup
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.max_seq_length = max_seq_length
        self.debug_steps = debug_steps
        self.do_pad = do_pad
        self.progress = progress
        self.num_labels = num_labels
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and not no_cuda else 'cpu'
        )
        self.default_tensor_type = torch.FloatTensor if self.device == 'cpu' else torch.cuda.FloatTensor
        torch.set_default_tensor_type(self.default_tensor_type)
        self.n_gpu = torch.cuda.device_count()
        self._set_random_seed()

        self.tokenizer = BertTokenizer.from_pretrained(
            bert_tokenizer_type,
            do_lower_case=do_lower_case,
        )
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def _set_random_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if not self.no_cuda:
            torch.cuda.manual_seed_all(self.seed)

    def _create_examples(self, lines, action_type):
        examples = []
        for idx, line in enumerate(lines):
            guid = '{ACTION_TYPE}-{IDX}'.format(
                ACTION_TYPE=action_type, IDX=idx)
            text_a = str(line[0])
            labels = line[1]
            examples.append(InputExample(
                guid=guid,
                text_a=text_a,
                text_b=None,
                labels=labels,
            ))
        return examples

    def _convert_examples_to_features(self, examples, output_mode):
        features = []
        max_length = 0
        for idx, example in enumerate(examples):
            if idx % 10000 == 1:
                logger.info('Writing example %d of %d'.format(
                    idx, len(examples)))

            tokens_a = self.tokenizer.tokenize(example.text_a)
            max_length = max(max_length, len(tokens_a))
            tokens = ['[CLS]'] + tokens_a[:self.max_seq_length - 2] + ['[SEP]']
            segment_ids = [0 for _ in range(len(tokens))]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 for _ in range(len(input_ids))]

            if self.do_pad:
                padding = [0 for _ in range(
                    self.max_seq_length - len(input_ids))]
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

            label_ids = example.labels
            features.append(InputFeature(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
            ))

        print(max_length)
        return features

    def _get_pretrained_model(self):
        self.model = BertForSequenceClassification.from_pretrained(
            self.bert_tokenizer_type,
            num_labels=self.num_labels,
        )
        self.model.to(self.device)

    def _get_optimizer_and_scheduler(self, num_training_steps):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.01,
            },
            {
                'params': [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.00,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            correct_bias=False,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup * num_training_steps,
            num_training_steps=num_training_steps,
        )

    def _generate_tensor_data(self, features):
        all_input_ids = torch.tensor(
            [feature.input_ids for feature in features],
        )
        all_input_mask = torch.tensor(
            [feature.input_mask for feature in features],
        )
        all_segment_ids = torch.tensor(
            [feature.segment_ids for feature in features],
        )
        all_label_ids = torch.tensor(
            [feature.label_ids for feature in features],
        )
        return TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids
        )

    def _fine_tune(self, features):
        num_training_steps = int(
            len(features) / self.batch_size / self.gradient_accumulation_steps
        ) * self.num_epochs
        logger.info('***** Running Fine Tuning Step *****')
        logger.info('Num examples = %d'.format(len(features)))
        logger.info('Batch size = %d'.format(self.batch_size))
        logger.info('Num steps = %d'.format(num_training_steps))

        data = self._generate_tensor_data(features)

        sampler = RandomSampler(data)
        dataLoader = DataLoader(
            data,
            sampler=sampler,
            batch_size=self.batch_size,
        )

        self._get_optimizer_and_scheduler(
            num_training_steps=num_training_steps,
        )

        self.model.train()

        global_step = 0
        nb_tr_steps = 0

        for _ in trange(int(self.num_epochs), desc='Epoch', disable=not self.progress):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(dataLoader, desc='Iteration', disable=not self.progress)):
                batch = tuple(e.to(self.device) for e in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(
                    input_ids,
                    segment_ids,
                    input_mask,
                    labels=None,
                )[0]

                loss_fct = BCEWithLogitsLoss()

                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    label_ids.view(-1,
                                   self.num_labels).type(self.default_tensor_type),
                )
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                if self.debug_steps > 0 and global_step < self.debug_steps:
                    print(loss)

    def fit(self, x, y):
        tn_examples = [list(p) for p in zip(x, y)]
        examples = self._create_examples(tn_examples, 'tn')
        tn_features = self._convert_examples_to_features(
            examples,
            self.output_mode,
        )
        self._get_pretrained_model()
        print('Start fine-tuning model')
        self._fine_tune(tn_features)
        print('Finish fine-tuning model')

    def _predict_prob(self, features):
        data = self._generate_tensor_data(features)

        sampler = SequentialSampler(data)
        dataLoader = DataLoader(
            data,
            sampler=sampler,
            batch_size=self.batch_size,
        )

        self.model.eval()

        probs = []
        for batch in tqdm(dataLoader, desc='Evaluating', disable=not self.progress):
            batch = tuple(e.to(self.device) for e in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                logits = self.model(
                    input_ids,
                    segment_ids,
                    input_mask,
                    labels=None,
                )[0]

            probs.append(logits.detach().cpu().sigmoid())

        return np.vstack(probs)

    def predict_prob(self, x, threshold=0.5):
        tt_examples = [[xx, 0] for xx in x]
        examples = self._create_examples(tt_examples, 'tt')
        tt_features = self._convert_examples_to_features(
            examples,
            self.output_mode,
        )
        return self._predict_prob(tt_features)

    def predict(self, x, threshold=0.5):
        probs = self.predict_prob(x)
        return (probs > threshold).astype(np.int)

    def load_model(self):
        self.model = BertForSequenceClassification.from_pretrained(
            MODEL_FOLDER,
            num_labels=self.num_labels,
        )
        self.model.to(self.device)

    def save_model(self):
        self.model.save_pretrained(MODEL_FOLDER)
