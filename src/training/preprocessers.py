# Preprocessing function for XSUM dataset
def xsum_preprocess_function(examples, tokenizer):
    inputs = []
    targets = []
    for i in range(len(examples['document'])):
        if examples['document'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['document'][i])
            targets.append(examples['summary'][i])


    inputs = ["summarize: " + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=1024, padding="max_length", truncation=True)

    labels = tokenizer(targets, max_length=50, padding="max_length", truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs