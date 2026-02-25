import os
import sys
from pypdf import PdfReader
import time
import json
import arxiv
from pathlib import Path
from vllm import LLM, SamplingParams
import subprocess
import json

def download_papers_by_query(query, max_results, download_dir='pdfs/'):
    # Create the download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    client = arxiv.Client()
    for result in search.results():
        try:
            title = result.title.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filepath = os.path.join(download_dir, f"{title}.pdf")
            print(f"Downloading: {result.title}")
            result.download_pdf(dirpath=download_dir, filename=f"{title}.pdf")
            print(f"Saved to {filepath}")
        except Exception as e:
            print(f"Error downloading {result.title}: {e}")

def get_docs(path_to_file, filenames): # adjust to read certain amount of files from a directory, x number of pages/docs at a time, PASS LIST OF PDF FILE NAMES

    #:param path_to_file: path to folder of pdfs
    #:return: filtered documents from that folder

    parsed = []
    hash_file_name = 'hash2.txt'
    pdf_files = [file.strip() for file in filenames if file.strip().lower().endswith('.pdf')]
    for file in pdf_files:
        # print(file)
        is_already_present = False
        try:
            doc = PdfReader(path_to_file+file, strict = False)
        except:
            print("Non-readable format")
            continue
        for index, page in enumerate(doc.pages):
            content = str(page.extract_text())

            if index==0: #if it's the first page
                content = content.replace("\n", "")
                content = content.replace(" ", "") # filter to one line
                if 'abstract' in content.lower():
                    content = content[0:content.lower().find('abstract')]
                with open(hash_file_name, 'r', encoding='utf-8') as f:
                    if (content + '\n') in f.readlines():
                        is_already_present = True
                    else:
                        with open(hash_file_name, 'a', encoding = 'utf-8') as f:
                            f.write(f"{content}\n")

            if is_already_present==False:
                parsed.append({file:content})
            else:
                break


    return parsed

def get_dump(docs):

    model_id = "NousResearch/Llama-2-7b-chat-hf"

    q_a_unfiltered = []

    print()
    prompts = []
    for text in docs:
        prompt = f"You are a professor who has to create a quiz with as many questions as possible on the following context : \n{list(text.values())[0]}.\n The title of the text is {list(text.keys())[0]} " \
        	     f"Please follow these requirements:\n" \
        	     f"1. Generate questions and answers based solely on the provided context.\n" \
      	         f"2. For each relationship, pattern, or dependency identified in the text, generate questions that explore these connections in a logical and analytical manner. For example, if the context states that 'X increases when Y increases in context of Z', frame questions like 'Why does X increase when Y increases in context of Z?' or 'What implications happen when X increases in context of Z? or 'How is X increasing in the given condition/context?'. Clarify what 'X','Y' are using proper nouns, and what the contition is .\n" \
                 f"3. For each piece of content in the text, create questions that abstracts the specific details into a more general concept. Do not ask specific paper questions like <How X is changing in above paper or text?> instead ask <How X changes in the given condition ?>  and clearly state the condition in the question.\n " \
                 f"4. Avoid simple or fact-based questions; instead ask 'Why', 'What', and 'How' questions that talks about the implications, causes, downsides, and benefits. \n" \
                 f"5. Avoid asking questions about the 'main idea', 'purpose of the text', 'authorial' details of the paper. Do not use paper specific question or answer formats like 'In this work...','in the given text','in the given code', 'Who is the author?, 'What is the main idea of this paper?', '....in this text?'.\n" \
                 f"6. Ensure questions are open-ended and drive towards critical thinking about the issues, problems, and methodologies mentioned. For example, if 'X' is a problem in the given context', make questions like 'What causes the issue with X, and how can it be resolved?'. if 'Y' is a solution to any given problem, ask questions like 'Why should Y be preferred over other solutions?' or 'How does Y help to improve the performance problem 'X'?' \n" \
                 f"7. Provide detailed, multi-sentence answers to each question that thoroughly explain the 'What','Why' and 'How' questions in a comprehensive manner.\n" \
                 f"8. Avoid generating questions with placeholders like X , Y, Z. Do not ask questions 'Why does X increase when Y increases?' and 'Why does the increase in Y affect the performance of the system?'.\n" \
                 f"9.  Use specific terms from the text and ensure your questions are clear and directly related to the context.\n" \
                 f"10. If there are any code examples in the context, generate questions related to the purpose, structure, and potential improvements of the code. Make sure to give details about the code and do not ask as 'in the given code'.\n" \
                 f"11. Make sure no questions are repeated and duplicated. \n" \
                 f"12. Format your output in JSON with the following structure: ['instruction': <generalized question>, 'keywords': <relevant keywords from the text>, 'output': <short, detailed answer>, 'context': <Provide a 5-8 sentences thorough and detailed explanation of context/terms needed for understanding the question-answer pair. This should include definitions, relevant theories, or frameworks, and any necessary background information that informs the inquiry without stating the importance of understanding the content. Don't say anything like 'In the given text', 'The text suggests that..', just explain the concept. >,  'source':{list(text.keys())[0]}].\n" \

        prompts.append(prompt)
        
    llm = LLM(
    model_id,
    dtype="float32",  # safer for CPU; avoids float16/bfloat16 errors
    max_model_len=2048)

    outputs = llm.generate(prompts, sampling_params = SamplingParams(temperature = 0.8, top_p = 1, max_tokens=1500))

    for output in outputs:
        generated_text = output.outputs[0].text
        q_a_unfiltered.append(generated_text)
    return q_a_unfiltered

def to_json(output_list, docs):

    #:param dictionary: input dictionary to convert to JSON
    #:return: None, simply outputs into 'out.json' file

    q_a_list= []
    for output in output_list:
        initial_split = output.split('{')
        for index, i in enumerate(initial_split):
            if index > 0:
                chosen = i.split('}')[0]
                chosen = chosen.replace("\n", "")
                chosen = chosen.strip()
                #print(chosen)
                #print()
                try:
                    if chosen[len(chosen)-1] =='"':

                        q_a_list.append(json.loads('{'+chosen+'}'))
                except:
                    continue
    filtered = []
    for elem in q_a_list:
        #print("Unfiltered: " + str(elem))
        present = False
        for doc in docs:
            if ('keywords' in elem.keys() and elem['keywords'] is not None):
                try:
                    #if elem['keyword'].lower() in list(doc.values())[0].lower().replace('\n', ''):
                        #present = True
                    keywords = elem['keywords']
                    present = any(word.lower() in list(doc.values())[0].lower().replace('\n', '') for word in keywords)

                except:
                    print(f"Problematic Keyword: {elem['keywords']}")
        if ('instruction' in elem.keys() and 'keywords' in elem.keys() and 'output' in elem.keys() and 'source' in elem.keys() and 'context' in elem.keys()) and ("" not in elem.values() and None not in elem.values() and '?' in elem['instruction']) and (not ('no information' in elem['output'] or 'not covered' in elem['output']) and present == True):
            #print(elem['keyword'])
            filtered.append(elem)
    filtered = [i for n, i in enumerate(filtered)
                if i not in filtered[:n]]
    result = '['
    for elem in filtered:
        result += str(elem)
    result += ']'
    print(result)
    with open('jsons/arxiv_version2.json', 'a') as f:
        json.dump(filtered, f)

if __name__ == '__main__':

    os.system('rm hash2.txt')
    with open('hash2.txt', 'w', encoding='utf-8') as f:
        pass
    os.system('rm -rf pdfs/')
    download_papers_by_query('Deep Learning', 20, 'pdfs/')

    start_python = time.time()
    parsed = get_docs('pdfs/', os.listdir('pdfs/'))
    end_python = time.time()
    python_time = end_python - start_python

    start_cpp = time.time()
    c_parsed_raw = subprocess.run(['./get_docs'], capture_output=True, text=True) # get_docs obtained from C++ code
    docs = json.loads(c_parsed_raw.stdout.strip()) # this mimics the return value of the original get_docs function
    end_cpp = time.time()
    cpp_time = end_cpp - start_cpp

    output = get_dump(docs)

    print("C++ time get_docs: ", cpp_time)
    print("Python time get_docs: ", python_time)

    if len(output)> 0:
        start_cpp
        with open('temp_output_list.json', 'w') as f:
            json.dump(output, f)

        # Save docs
        docs_content = []
        for entry in parsed:
            for key, val in entry.items():
                docs_content.append(val)

        with open('temp_docs.json', 'w') as f:
            json.dump(docs_content, f)

        # call C++ executable
        subprocess.run(['./to_json', 'temp_output_list.json', 'temp_docs.json']) #to_json obtained from C++ code
        cpp_time = time.time() - start_cpp

        start_python = time.time()
        to_json(output, docs)
        end_python = time.time()
        python_time = end_python - start_python
        
        print("C++ time to_json: ", cpp_time)
        print("Python time to_json: ", python_time)

    else:
        print("No documents found.")

    
