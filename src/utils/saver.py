import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def save_matrixes(output_path, matrix2save, id_column=[], rename_cols=[]):
    df = pd.DataFrame(matrix2save)
    if(id_column!=[]):
        df["ids"] = id_column
    if(rename_cols!=[]):
        for i in range(0,len(rename_cols)):
            df.columns.values[i] = rename_cols[i]
    df.to_csv(output_path, index=False, sep=";")
    return df


def save_heatmap_from_matrix(input_path_csv, output_path_png, epsilons, min_samples, format=".4f",
                               figsize=(15, 10)):
    """
       Create images as heatmaps from the csv matrix passed in input_path_csv
           Args:
            input_path_csv (str): Input path where we can find the matrix with the data to plot
            output_path_png (str): Output path where we want to save the generated heatmaps
            epsilons (list): List with the epsilon values referenced in the matrix as the rows
            min_samples (list): List with the min_samples values referenced in the matrix as the columns
            format (str-format): Format for representing the data of the matrix
            figsize (tuple(int,int)): Size of the image to save
    """
    df = pd.read_csv(input_path_csv, names=min_samples)
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df, annot=True, fmt=format)
    heatmap.yaxis.set_ticklabels(epsilons, rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(min_samples, rotation=0, ha='right')
    plt.ylabel('epsilon')
    plt.xlabel('min_samples')
    # fig.show()
    plt.savefig(output_path_png)
    plt.close(fig)

