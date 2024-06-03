import json
import sys

folder = sys.argv[1]

# open file in read-mode
filename = "./" + folder + "/result.json"
with open(filename, "r") as file:
    # read JSON data
    data = json.load(file)
    print(data["images"])
    for image in data["images"]:
        print("\n {}".format(image["file_name"]))
        edit_fname = image["file_name"].split("/")

        # Update the value
        image["file_name"] = edit_fname[-1]

# Open the file for writing and write the updated data
with open(filename, "w") as f:
    json.dump(data, f, indent=4)
