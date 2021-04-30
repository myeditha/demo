from transformers import pipeline, set_seed
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # add the EOS token as PAD token to avoid warnings
    model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

    tf.random.set_seed(42)
    line = ""

    while(line != "quit()"):
        line = input("Please enter a prompt: ").rstrip()

        input_ids = tokenizer.encode(line, return_tensors='tf')

        sample_outputs = model.generate(
            input_ids,
            do_sample=True, 
            max_length=100, 
            top_k=100, 
            top_p=0.95, 
            num_return_sequences=3
        )

        print("Output:\n" + 100 * '-')
        for i, sample_output in enumerate(sample_outputs):
            print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

