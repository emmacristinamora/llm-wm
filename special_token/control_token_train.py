import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

NEW_TOKEN = "<st>"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def resize_vocab(model, tokenizer):
    tokenizer.add_tokens([NEW_TOKEN])
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

def set_grads(model):
    for param in model.parameters():
        param.requires_grad = False
    
    model.embed_tokens.weight.requires_grad = True

def set_loss_hook(model, tokenizer):
    embedding_layer = model.embed_tokens
    mask = torch.zeros(len(embedding_layer.weight), 1, device=DEVICE, dtype=torch.bfloat16) # dtype to match Qwen default, otherwise RunTimeError hook '<lambda>' has changed the type of value (was CUDABFloat16Type got torch.cuda.FloatTensor)
    mask[len(tokenizer) - 1] = 1.0
    hook = embedding_layer.weight.register_hook(lambda grad: grad * mask)
    return hook

def train_step(model, optimizer, inputs):

    outputs = model(**inputs)

    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def validate(model, val_inputs):
    with torch.no_grad():
        outputs = model(**val_inputs)
        return outputs.loss.item()

def create_inputs(tokenizer, conversation, idx):

    prompt_text = tokenizer.apply_chat_template(conversation[:idx-1], tokenize = False)
    prompt_text += NEW_TOKEN

    gt_text = f"<|im_start|>user\n{conversation[idx]["content"]}<|im_end|>\n"

    full_text = prompt_text + gt_text

    inputs = tokenizer(full_text, return_tensors = "pt").to(DEVICE)

    prompt_length = len(tokenizer(prompt_text)["input_ids"])
    labels = inputs["input_ids"].clone()
    labels[0, :prompt_length] = -100

    inputs["labels"] = labels
    return inputs

def main():

    # Load 
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Introduce new token
    resize_vocab(model, tokenizer)

    # Set required gradients and hooks
    set_grads(model.model)
    hook = set_loss_hook(model.model, tokenizer)

    sample_conversation = [{'content': 'Can you translate English to German?', 'role': 'user'},
    {'content': "Yes, I can certainly help you with that. Just provide the text you'd like translated into German.",
    'role': 'assistant'},
    {'content': ' Please translate Thank you into German',
    'role': 'user'},
    {'content': "Die Übersetzung für 'Thank you' ist 'Dankeschön'. Möchten Sie auch wissen, wie man förmlich 'Vielen Dank' sagt?",
    'role': 'assistant'},
    {'content': ' No thanks. What city should I visit in Germany as an Italian?',
    'role': 'user'},
    {'content': "You should definitely visit Munich, often called 'Italy's northernmost city' because of its Mediterranean flair and beautiful Italian-inspired architecture like the Odeonsplatz. Alternatively, Cologne is a fantastic choice for its Roman history and more open, 'warm' temperament that many Italian visitors find very welcoming.",
    'role': 'assistant'},
    {'content': 'Where would I find the cheaper coffee in general?',
    'role': 'user'},
    {'content': 'In general, you will find cheaper coffee in Cologne, where a cappuccino averages around €3.64 compared to over €3.80 in Munich. While Munich is the most expensive city in Germany, Cologne offers a more budget-friendly experience for socializing and dining out.',
    'role': 'assistant'},
    {'content': 'And in Munich I should probably visit the Oktoberfest then right, is it fun', 'role': 'user'},
    {'content': "It's legendary for its atmosphere and massive beer tents, though it only runs from late September to early October. If you enjoy lively crowds and Bavarian tradition, it's an absolute blast, but I can check the specific 2026 dates for you if you're planning a trip.",
    'role': 'assistant'}]

    # NOTE: For now assume fixed input and output, no batching

    train_inputs = create_inputs(tokenizer, sample_conversation, idx=-2)
    val_indices = [-4, -6, -8]
    val_inputs = [create_inputs(tokenizer, sample_conversation, idx) for idx in val_indices]
    
    train_steps = 100
    lr = 1e-3

    optimizer = torch.optim.AdamW([model.model.embed_tokens.weight], lr=lr)


    cum_loss = 0.0
    for step in range(train_steps):
        if step % 10 == 0:
            val_loss = dict()
            for i, val_input in zip(val_indices, val_inputs):
              val_loss[i] = validate(model, val_input)
            print(f"Step {step}: Cum Loss is {cum_loss:.2f} / Validation Loss is {val_loss}")
            cum_loss = 0.0

        cum_loss += train_step(model, optimizer, train_inputs)



if __name__ ==  "__main__":
    main()