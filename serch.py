import glob

path = "C:\python"

for file in glob.glob(path + "\*.py"):
    print(file)