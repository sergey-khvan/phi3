import os
import functools
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    # get_scheduler,
)
from transformers.models.phi.modeling_phi import PhiDecoderLayer
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    
def cleanup():
    dist.destroy_process_group()
    

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        # lr_scheduler,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.train_data = train_data
        self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        self.snapshot_path = snapshot_path
        self.save_every = save_every
        self.epochs_run = 0
        self.init_start_event = torch.cuda.Event(enable_timing=True)
        self.init_end_event = torch.cuda.Event(enable_timing=True)
        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                PhiDecoderLayer, # FIXME: change to Phi3DecoderLayer, when it will be available (same in imports)
            },
            )
        self.model = model.to(self.gpu_id)
        self.model = FSDP(self.model, auto_wrap_policy=my_auto_wrap_policy)

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        self.ddp_loss = torch.zeros(1).to(self.gpu_id)
        for batch in self.train_data:
            batch = {k: v.to(self.gpu_id) for k, v in batch.items()}
            self.optimizer.zero_grad()
            outputs = self.model(**batch) #FIXME
            loss = outputs.loss
            loss.backward()

            self.optimizer.step()
            self.ddp_loss[0] += loss.item()
        dist.all_reduce(self.ddp_loss, op=dist.ReduceOp.SUM)
        if self.gpu_id == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, self.ddp_loss[0]))

    def train(self, max_epochs: int):
        self.init_start_event.record()
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        self.init_end_event.record()

def load_train_obj():
    def preprocess_function(examples):
        return tokenizer(examples["question"], examples["answer"])

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    base_model = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dataset = load_dataset("ruggsea/stanford-encyclopedia-of-philosophy_instruct")

    # dataset = dataset["train"]
    # dataset = dataset.train_test_split(test_size=0.1)

    tokenized_ds = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
    )

    block_size = 128
    lm_dataset = tokenized_ds.map(group_texts, batched=True, num_proc=1)

    tokenizer.pad_token = tokenizer.eos_token
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    return model, lm_dataset, tokenizer, optimizer


def fsdp_main(save_every, num_epochs, snapshot_path: str = "snapshot.pt"):
    setup()
    batch_size = 16
    model, lm_dataset, tokenizer, optimizer = load_train_obj()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        lm_dataset["train"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        sampler=DistributedSampler(dataset=lm_dataset["train"]),
    )
    
    trainer = Trainer(model, train_dataloader, optimizer, save_every, snapshot_path)
    trainer.train(num_epochs)
    cleanup()
    
if __name__ == '__main__':
    num_epochs = 3
    save_every = 1
    fsdp_main(save_every, num_epochs)