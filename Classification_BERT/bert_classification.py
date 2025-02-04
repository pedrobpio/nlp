from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# this is the class to perform the BERT classification task
class BERT_like_classification():

    def __init__(self, model_name = "google-bert/bert-base-uncased") -> None:
        """
        Initializes the class with empty attributes to store data and models.
        """
        self.ds = None
        self.model = None
        self.tokenized_ds = None
        self.tokenizer = None
        self.model_name = model_name
        self.id2label = None
        self.label2id = None
        self.ds_splits = None
        self.data_collator = None
        self.trainer = None
        self.training_args = TrainingArguments(
            output_dir="bert-computers-model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="none"
        )


    def set_training_arguments(self, training_arguments):
        """
        Sets the training args for the Trainer
        """
        self.training_args = training_arguments

    def load_model(self, model_name):
        """
        Loads the model with AutoModelForSequenceClassification method.
        you have to first load the dataset, otherwise it will not work properly.
        Args:
            model_name: Name of the model that we will load
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(self.id2label), id2label=self.id2label, label2id=self.label2id)

    def set_tokenizer(self, model_name):
        """
        Sets the tokenizer with AutoTokenizer.from_pretrained method
        Args:
            model_name: Name of the model that we will load the tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def set_data_collator(self):
         """
        Sets the data Colator, it has to be executed after the set_tokenizer to work.
        """
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    
    def load_dataset(self, data_file):
        """
        Loads the dataset to the class and create the id2label and label2id variables.
        Args:
            data_file: the data file directory that we will use, it shoud be a .CSV to work
        """
        self.ds = load_dataset("csv", data_files=data_file)
        class_set = set(self.ds['train']['class'])
        self.id2label = {i: cls for i, cls in enumerate(class_set)}
        self.label2id = {cls: i for i, cls in enumerate(class_set)}

    def split_dataset(self):
        """
        Splits the dataset into 70-10-20.
        """
        ds_train_devtest = self.ds['train'].train_test_split(test_size=0.3, seed=42)
        ds_devtest = ds_train_devtest['test'].train_test_split(test_size=0.66, seed=42)

        self.ds_splits = DatasetDict({
            'train': ds_train_devtest['train'],
            'valid': ds_devtest['train'],
            'test': ds_devtest['test']
        })


    def preprocess_function(self, examples):
        """
        Function to preccess the tokenizer
        """
        return self.tokenizer(examples["text"], truncation=True)
    

    def tokenize_ds_splits(self):
        """
        method that applies the tokenizer into the splited datasets.
        """
        self.tokenized_ds = self.ds_splits.map(self.preprocess_function, batched=True)
        
        # add labels to the tokenizer_ds variable
        self.tokenized_ds = self.tokenized_ds.map(
            lambda example: {"labels": self.label2id[example["class"]]})
    

    def compute_metrics(self, predictions):
        """
        method that compute the metrics during the training and the trainer predictions.
        Args:
            predictions: list of prediction of the model.
        """
        preds = np.argmax(predictions.predictions, axis=1)
        return {
            "accuracy": accuracy_score(predictions.label_ids, preds),
            "f1_micro": f1_score(predictions.label_ids, preds, average="micro"),
            "f1_macro": f1_score(predictions.label_ids, preds, average="macro"),
            
        }

    def train_model(self,train_dataset, eval_dataset,  ):
        """
        Train the model. it uses the Trainer class to train the model, you can configure the arguments using set_training_arguments method
        you also have to have executed created the model, tokenizer and data colator so that it can be called.
        the method will ask for a training dataset and a eval dataset.
        Args:
            train_dataset: dataset that will be used to train the model
            eval_dataset: dataset used to evaluate the model.
        """
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()
    
    def model_predict(self,test_dataset):
        """
        method that uses the Trainer to perform the model prediction on a dataset.
        Args:
            test_dataset: dataset that will be used to test the model
        """
        return self.trainer.predict(test_dataset)