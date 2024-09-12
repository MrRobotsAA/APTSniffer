import json
import openai
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from concurrent.futures import ThreadPoolExecutor, as_completed

openai_api_key = 'xx'
client = openai.OpenAI(api_key=openai_api_key)


def calculate_total_cost_rmb(total_input_tokens, total_output_tokens, gpt4=False, exchange_rate_usd_to_rmb=1/0.14):
    
    if gpt4:  #gpt-4o or gpt-4o-2024-05-13
        cost_per_token_input = 5.00 / 1000000  
        cost_per_token_output = 15.00 / 1000000  
    else:  # gpt-3.5-turbo-0125 or gpt-3.5-turbo-instruct
        cost_per_token_input = 0.50 / 1000000  
        cost_per_token_output = 1.50 / 1000000  

）
    total_cost_usd = (total_input_tokens * cost_per_token_input) + (total_output_tokens * cost_per_token_output)


    total_cost_rmb = total_cost_usd * exchange_rate_usd_to_rmb

    return total_cost_rmb

def classify_text(prompt):
    formatted_prompt = f"You are an experienced APT traffic analysis expert. Please classify the following traffic support data into one of the categories below based on the reference information provided.\nText: {prompt}\nCategories: Label0_Benign_Traffic, Label1_APT_Traffic"
    functions = [
        {
            "name": "classifyText",
            "description": "Classify following data into categories and provide probabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of the text."
                    },
                    "probabilities": {
                        "type": "object",
                        "properties": {
                            "Label0_Benign_Traffic": {"type": "number"},
                            "Label1_APT_Traffic": {"type": "number"}
                        }
                    }
                },
                "required": ["category", "probabilities"]
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ],
        functions=functions,
        function_call={"name": "classifyText"},
        temperature=0.3,
        max_tokens=1024,
        top_p=0.5,
        frequency_penalty=0.1,
        presence_penalty=0.1
    )

    function_call_arguments = response.choices[0].message.function_call.arguments
    result = json.loads(function_call_arguments)
    classification = result.get("category")
    probabilities = result.get("probabilities", {})


    function_call_arguments = response.choices[0].message.function_call.arguments
    result = json.loads(function_call_arguments)
    classification = result.get("category")
    probabilities = result.get("probabilities", {})


    money1 = response.usage.prompt_tokens
    money2 = response.usage.completion_tokens
    print("money1,money2:",money1,money2)
    total_money = calculate_total_cost_rmb(money1, money2, False)
    total_money2 = calculate_total_cost_rmb(money1, money2, True)
    return classification, probabilities, total_money


def generate_prompts_en(merged_samples, list_ans):
    prompts = []
    for key, sample in merged_samples.items():
        prompt = f"Sample ID: {key}\n"
        prompt += "Exact Match:\n"
        for k, v in sample[0].items():
            k = int(k)
            prompt += f"  Label {k} count distribution: {v}\n"
            prompt += f"  Recommended training weight: {list_ans[0][k]}\n"
        prompt += "Fuzzy Match:\n"
        for k, v in sample[1].items():
            k = int(k)
            prompt += f"  Label {k} count distribution: {v}\n"
            prompt += f"  Recommended training weight: {list_ans[1][k]}\n"
        prompt += "Relational Match:\n"
        for k, v in sample[2].items():
            k = int(k)
            prompt += f"  Label {k} count distribution: {v}\n"
            prompt += f"  Recommended training weight: {list_ans[2][k]}\n"
        prompts.append(prompt)
    return prompts


def convert_label(label):
    if label == "Label0_Benign_Traffic":
        return 0
    elif label == "Label1_APT_Traffic":
        return 1
    else:
        return -1

def query_llm_function():
    with open('./data/test_labels_Earlyflow.txt', 'r') as f:
        test_labels = eval(f.read())

    with open('./data/merged_samples_test_Earlyflow.txt', 'r') as f:
        merged_samples_test = eval(f.read())

    with open('./data/list_ans_Earlyflow.txt', 'r') as f:
        list_ans = eval(f.read())

    prompts = generate_prompts_en(merged_samples_test, list_ans)
    print(type(prompts[0]))
    print(type(prompts))
    print(prompts[0])

    results = []
    from tqdm import tqdm
    total_money = 0
    for prompt in tqdm(prompts):
    # for prompt in prompts:
        classification, probabilities,money1 = classify_text(prompt)
        total_money += money1
        print("gpt3.5 total_money:", total_money,"元")
        results.append(classification)


 
    results = [convert_label(label) for label in results]
    print("results:", results)

  
    with open('./data/gpt3.5_earlyflow_results.txt', 'w') as f:
        f.write(str(results))


    with open('./data/gpt3.5_earlyflow_results.txt', 'r') as f:
        results = eval(f.read())

 
    if len(results) < len(test_labels):
        results += [results[-1]] * (len(test_labels) - len(results))

  
    test_labels = list(test_labels.values())
    # print("test_labels:", test_labels)
    print("len(test_labels):", len(test_labels))

 
    y_true = test_labels
    y_pred = results

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

if __name__ == "__main__":
    query_llm_function()



