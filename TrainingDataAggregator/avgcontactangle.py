import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('houghdir')
parser.add_argument('mldir')
parser.add_argument('output')
args = vars(parser.parse_args())

def calculate_average_contact_angle(directory):
    """Calculates the average 'Contact Angle' for each CSV file in the given directory."""
    averages = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)
                if "Contact Angle" in df.columns:
                    avg_contact_angle = df["Contact Angle"].mean()
                    averages[filename] = avg_contact_angle
                else:
                    print(f"Warning: 'Contact Angle' column not found in {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return averages

avg_dir1 = calculate_average_contact_angle(args['houghdir'])
avg_dir2 = calculate_average_contact_angle(args['mldir'])

# Combine results into a single DataFrame
results = pd.DataFrame({
    "Filename": list(avg_dir1.keys()) + list(avg_dir2.keys()),
    "Directory": ["Hough Transform"] * len(avg_dir1) + ["ML"] * len(avg_dir2),
    "Average Contact Angle": list(avg_dir1.values()) + list(avg_dir2.values())
})

# Save results to a CSV file
results.to_csv(args['output'], index=False)