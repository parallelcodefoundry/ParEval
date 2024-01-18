#!/bin/python3

from argparse import ArgumentParser
import json
import os
from string import Template

# READ IN ARGUMENTS
parser = ArgumentParser(description=__doc__)
parser.add_argument("prompts", help="Path to prompts.json file")
parser.add_argument("driver_num", help="Numbers of drivers to create outputs for",
                    nargs="+", type=int)
parser.add_argument("output_file", help="Output .json filename")
args = parser.parse_args()

# READ IN PROMPTS
with open(args.prompts, "r") as f:
    prompts = json.load(f)

prompts = list(filter(lambda x: any([str(y) in x["name"] for y in args.driver_num]), prompts))

outputs = []

for p in prompts:
    new_entry = p
    new_entry["outputs"] = ["undefinedFunctionCall(); }"]

    last_line = "".join(p["prompt"].splitlines()[-1:]).split(" ")
    ret_type = last_line[0]
    if (ret_type == "__global__"):
        ret_type = last_line[1]
    print(ret_type)
    if (ret_type == "void"):
        new_entry["outputs"].append("}")
    elif (ret_type == "int" or ret_type == "size_t"):
        new_entry["outputs"].append("return 0;}")
    else:
        new_entry["outputs"].append("return 0.0;}")

    outputs.append(new_entry)

with open(args.output_file, "w") as f:
    f.write(json.dumps(outputs, indent=4))
