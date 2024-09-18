import os
import sys

counter = 0

def read_dir(dir, outfile):
    global counter
    print(dir)
    # Iterate through all files in the directory
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if(os.path.isdir(file_path)):
            read_dir(file_path, outfile)
        # Check if the filename starts with "WP" and ends with ".mp4"
        if filename.startswith("WP") and filename.endswith(".mp4"):
            # Get the absolute path of the file

            # Add the absolute path to the output text file
            with open(outfile, 'a') as fp:
                fp.write(file_path + '\n')
                counter += 1


# The directory where your files are located
dir = sys.argv[1]

# The name of the output text file
outfile = "wp_mp4_paths.txt"

read_dir(dir, outfile)

print(counter)
print(os.getcwd() + "\\wp_mp4_paths.txt")