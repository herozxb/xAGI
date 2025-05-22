import httpx
import re
import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
import ollama


#"content": "Write a Python show a web page with just 1 image from web url = 'https://www.google.com/images/logo.gif', make the picture in the center of page, and add a input area and a botton follow it",


response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "user",
            "content": "Write a Python to show a rotating 3D flower in real movetion",
        },
    ],
)
#print(response["message"]["content"])

output = response["message"]["content"]

print("============generated_code================")
print(output)
    
def extract_python_code(text):
    # Regex to match code blocks within the text
    matches = re.findall(r'```python\n([\s\S]+?)\n```', text) # `r'```python\n([\s\S]+?)\n```'`
    return matches

code_blocks = extract_python_code(output)
#print(code_blocks)

def_code = []
non_def_code = []

for block in code_blocks:
    if 'def ' in block:
        def_code.append(block)
    else:
        non_def_code.append(block)

print("==========[Function Definition Code][1]===========\n")
for code in def_code:
    print(code)
    print("==========[autorun_python_code]===========\n")
    exec(code)

print("==========[Non-Function Code][2]===========\n")
for code in non_def_code:
    print(code)
    print("==========[autorun_python_code]===========\n")
    exec(code)
    
    
    
    
# reference code    
'''
print("==========[1]===========\n")
# Save the generated code to a file
with open("multiply2.py", "w") as f:
    f.write(generated_code)
    
print("==========[2]===========\n")
# Optionally, execute the generated code
exec(generated_code)


print("==========[3]===========\n")
# Extract the function name using a regex
def get_function_name(function_def):
    match = re.search(r'def (\w+)\(', function_def)
    if match:
        return match.group(1)
    return None

# Get the function name from the definition string
function_name = get_function_name(generated_code)

if function_name:
    print(f"Function name extracted: {function_name}")
    
    # Call the function dynamically using the function name
    result = globals()[function_name](15, 15, 15)
    print(f"Output of {function_name}(15, 15, 15): {result}")
else:
    print("Function name could not be extracted.")
''' 

