from transformers import pipeline, set_seed
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
import math


def group_texts(examples):
    # Concatenate all texts.
    print(examples.keys())
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

if __name__ == "__main__":
    # datasets = load_dataset("csv", data_files={"train": "archive/stories.train.csv", "validation": "archive/stories.test.csv"})
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # print(datasets["train"][10])

    # tokenized_datasets = datasets.map(lambda x: tokenizer(x["content"]), batched=True, num_proc=4,remove_columns=["content"])
    # print(tokenized_datasets["train"][10])

    # tokenized_datasets.save_to_disk("./archive-tokenized")
    # model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

    model_checkpoint = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    tokenized_datasets = load_from_disk("./archive-tokenized")
    print("loaded datasets.")
    tokenized_datasets = tokenized_datasets.remove_columns(["bookno"])
    block_size = tokenizer.model_max_length
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=lm_datasets["train"],         # training dataset
        eval_dataset=lm_datasets["validation"]            # evaluation dataset
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.save_model("./model/init")

    # # add the EOS token as PAD token to avoid warnings

    # tf.random.set_seed(42)
    # line = ""

    # training_args = TrainingArguments(
    #     output_dir='./results',          # output directory
    #     num_train_epochs=3,              # total # of training epochs
    #     per_device_train_batch_size=16,  # batch size per device during training
    #     per_device_eval_batch_size=64,   # batch size for evaluation
    #     warmup_steps=500,                # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,               # strength of weight decay
    #     logging_dir='./logs',            # directory for storing logs
    # )

    # trainer = Trainer(
    #     model=model,                         # the instantiated 🤗 Transformers model to be trained
    #     args=training_args,                  # training arguments, defined above
    #     train_dataset=train_dataset,         # training dataset
    #     eval_dataset=test_dataset            # evaluation dataset
    # )
    # print("\n\n")
    # while(line != "quit()"):
    #     line = input("Please enter a prompt: ").rstrip()

    #     input_ids = tokenizer.encode(line, return_tensors='tf')

    #     sample_outputs = model.generate(
    #         input_ids,
    #         do_sample=True, 
    #         max_length=100, 
    #         top_k=100, 
    #         top_p=0.95, 
    #         num_return_sequences=3
    #     )

    #     print("Output:\n" + 100 * '-')
    #     for i, sample_output in enumerate(sample_outputs):
    #         print("{}: {}\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

