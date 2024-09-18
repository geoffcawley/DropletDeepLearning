import os
import sys

# Input and output file paths
input_file = sys.argv[1]
output_dir = sys.argv[2]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

line_counter = 0
output_filename = "output.txt"

# Open the input file for reading and create a new output file
with open(input_file, 'r') as f_in, open(os.path.join(output_dir, output_filename), 'w') as f_out:
    for line in f_in:
        # Write the line to the output file
        f_out.write(line)

        # Increment the line counter and check if it's a multiple of 10
        line_counter += 1
        if line_counter % 10 == 0:
            # Close the current output file and create a new one for the next group of lines
            f_out.close()
            output_filename = f"output{line_counter//10}.txt"
            f_out = open(os.path.join(output_dir, output_filename), 'w')