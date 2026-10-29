import json, zipfile
r = json.loads(open(r"C:\Github\drone_training\v8\audit\v5_audit.json").read())
for k, v in r["issues"].items():
    print(k, v)
    z = zipfile.ZipFile(r"C:\Users\Nazar\Downloads\drone_dataset_v5.zip")
    print(z.read(k.replace("images/", "labels/").rsplit(".", 1)[0] + ".txt").decode())
