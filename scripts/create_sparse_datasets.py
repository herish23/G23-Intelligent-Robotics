## this is a script that i made to make the sparsity test
## the way it works is that we take the clean dataset and use our level and remove lidar value
## since we have all 359 then we can itereativeli loop it and remove per batch (skip to sparse level)
## do not run this untill and unless you dont have teh csv 
## i already have given the made csvs for sparsity inside the data folder 
## so you can directly import that csv on your model to test it 
import csv
import os
SPARSITY_LEVELS = [2, 4, 8, 16, 32]
INPUT_CSV = '../data/sensor_data_clean.csv'
OUTPUT_DIR = '../data/'

def create_sparse_dataset(input_path, output_path, sparsity):
    print(f"\ncreating sparse dataset with sparsity={sparsity}")
    print(f"input: {input_path}")
    print(f"output: {output_path}")

    with open(input_path, 'r') as fin:
        reader = csv.reader(fin)
        header = next(reader)

        ## find column indices
        ts_idx = header.index('timestamp')
        range_start = header.index('range_0')
        range_end = header.index('range_359')

        ## columns after lidar (v, w, gt_x, gt_y, gt_theta, etc.)
        other_cols_start = range_end + 1

        ## build new header with sparse lidar columns
        new_header = ['timestamp']

        ## keep every Nth ray
        rays_kept = 0
        for i in range(0, 360, sparsity):
            new_header.append(f'range_{i}')
            rays_kept += 1

        ## add remaining columns (v, w, ground truth)
        new_header.extend(header[other_cols_start:])

        print(f"original rays: 360")
        print(f"sparse rays: {rays_kept}")
        print(f"reduction: {rays_kept/360:.1%}")

        ## write sparse dataset
        with open(output_path, 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(new_header)

            row_count = 0
            for row in reader:
                new_row = [row[ts_idx]]  ## timestamp

                ## sparse lidar columns
                for i in range(0, 360, sparsity):
                    col_idx = range_start + i
                    new_row.append(row[col_idx])

                ## other columns (v, w, ground truth)
                new_row.extend(row[other_cols_start:])

                writer.writerow(new_row)
                row_count += 1

            print(f"wrote {row_count} rows")

def main():
    print("="*60)
    print("SPARSE DATASET GENERATOR")
    print("="*60)

    ## check input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"error: input file not found: {INPUT_CSV}")
        print("make sure you run this script from the scripts/ directory")
        return


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for sparsity in SPARSITY_LEVELS:
        output_file = os.path.join(OUTPUT_DIR, f'sensor_data_sparse_{sparsity}.csv')
        create_sparse_dataset(INPUT_CSV, output_file, sparsity)

    print("\n" + "="*60)
    print("COMPLETE!")


if __name__ == "__main__":
    main()
